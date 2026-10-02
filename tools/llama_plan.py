"""tools/llama_plan.py - run a model Strata's engine cannot (another architecture) with llama.cpp's expert offload,
planned for THIS PC from the GGUF header alone.

    python tools/llama_plan.py /models/Model-Q4_K_M.gguf                          (VRAM, RAM and context: detected / 32K)
    python tools/llama_plan.py owner/repo --match Q4_K_M --vram 16 --ram 32 --ctx 32768
    python tools/llama_plan.py /models/Model-Q4_K_M.gguf --run                    (starts llama-server on 127.0.0.1:8080)

Strata's own engine runs the Flash-Next layout (`qwen4exp`) only (tools/model_check.py says which side a file is on).
For any other architecture the same idea - attention and the dense layers on the GPU, the experts in RAM, as many
layers' experts on the GPU as the VRAM leaves room for - is llama.cpp's `-ngl` plus `--n-cpu-moe` (or `-ot`).  This
tool does the memory arithmetic that makes those numbers right for your VRAM / RAM / context, prints the plan, and with
--run starts `llama-server` on Strata's address, so a client keeps its URL: http://127.0.0.1:8080/v1.

What it is not: Strata's engine.  llama.cpp has no hot-expert cache on the GPU, no CPU/GPU overlap per layer and no MTP
draft head, so for the same model it is slower than Strata would be; and it can only run what llama.cpp implements.
The numbers come from the header: tensor sizes (the weights as stored), a KV cache from the attention keys (an upper
bound for hybrid models without a per-layer table), and the reserves you set.  --ram-bw adds a roofline ceiling for
the decode speed; the real speed is lower.

Exit status: 0 every plan ok/tight/slow, 1 at least one `no`, 2 a file could not be read.  -v / -vv log to stderr.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_check as M  # noqa: E402

log = logging.getLogger("llama_plan")

GIB = 1 << 30
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}              # bytes per cached element
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up|down|gate_up)_exps\.(?:weight|bias)$")
BLOCK = re.compile(r"^blk\.(\d+)\.")
OUTPUT = ("output.weight", "output_norm.weight", "output_norm.bias")    # llama.cpp offloads these with the last layer
EMBED = "token_embd.weight"                                             # the input embedding stays in RAM
ALL_LAYERS = 999                                                        # -ngl: more than any model has layers
SWA_UBATCH = 512                                                        # llama-server's -ub: a window cache holds the window + one micro-batch
SWA_PAD = 256                                                           # llama.cpp pads a KV cache to this many cells
DEFAULT_CTX = 32768
SERVER_NAMES = ("llama-server", "llama-server.exe")


# ------------------------------------------------------------------------------------------------ this PC
def detect_vram_gib():
    """Free VRAM of the first NVIDIA card in GiB, or None (no nvidia-smi, or it fails)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=15, check=True).stdout
        mib = [int(x) for x in out.split() if x.strip().isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        log.info("nvidia-smi failed: %s", e)
        return None
    if len(mib) > 1:
        log.warning("%d cards found: planning for the first (%.1f GiB free); llama.cpp splits layers over cards "
                    "by itself, pass --vram for the total you want it to use", len(mib), mib[0] / 1024)
    return mib[0] / 1024 if mib else None


def detect_ram_gib():
    """Installed RAM in GiB, or None."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MemoryStatus()
        m.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullTotalPhys / GIB
    return None


# ------------------------------------------------------------------------------------------------ the arithmetic
@dataclasses.dataclass
class Plan:
    target: str
    status: str = "no"                 # ok | tight | slow | no | error
    architecture: str | None = None
    model_path: str | None = None      # the first shard, when the target is a local file
    facts: dict = dataclasses.field(default_factory=dict)
    ngl: int | None = None
    n_cpu_moe: int | None = None
    flags: list = dataclasses.field(default_factory=list)       # llama-server's flags (not -m), --n-cpu-moe form
    flags_ot: list = dataclasses.field(default_factory=list)    # the same with -ot, for builds before --n-cpu-moe
    ceiling_tok_s: float | None = None
    problems: list = dataclasses.field(default_factory=list)
    notes: list = dataclasses.field(default_factory=list)


def _meta(meta: dict, arch: str, key: str):
    """An int, a list of ints (a per-layer table) or None."""
    v = meta.get(f"{arch}.{key}")
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return [int(x) for x in v] if isinstance(v, list) and v else None


def _scalar(v):
    """A per-layer table's largest entry, or the number itself."""
    return max(v) if isinstance(v, list) else v


def swa_flags(meta: dict, arch: str, n_layers: int):
    """Which layers use sliding-window attention: a list of bool, or None when the header does not say."""
    pat = meta.get(f"{arch}.attention.sliding_window_pattern")
    if isinstance(pat, list) and len(pat) == n_layers:
        return [bool(x) for x in pat]                       # a per-layer table: true = a sliding-window layer
    if isinstance(pat, int) and not isinstance(pat, bool) and pat >= 1:
        return [i % pat < pat - 1 for i in range(n_layers)]     # every pat-th layer is global (llama.cpp's default order)
    return None


def kv_layers(meta: dict, arch: str, n_layers: int, ctx: int, kv: str, parallel: int = 1):
    """(bytes of KV cache per layer, notes): K and V for `ctx` tokens at the cache type's width."""
    heads, kv_heads = _meta(meta, arch, "attention.head_count"), _meta(meta, arch, "attention.head_count_kv")
    width = _meta(meta, arch, "embedding_length")
    if kv_heads is None:
        kv_heads = heads
    if kv_heads is None:
        raise ValueError(f"no {arch}.attention.head_count_kv: pass --kv-gib")
    key_len = _scalar(_meta(meta, arch, "attention.key_length"))
    if key_len is None:
        h = _scalar(heads)
        if not (h and width):
            raise ValueError(f"no {arch}.attention.key_length and no head count to derive it: pass --kv-gib")
        key_len = width // h
    val_len = _scalar(_meta(meta, arch, "attention.value_length")) or key_len
    notes = []
    if isinstance(kv_heads, list):
        if len(kv_heads) != n_layers:
            raise ValueError(f"{arch}.attention.head_count_kv has {len(kv_heads)} entries for {n_layers} layers")
        per_layer = kv_heads
    else:
        per_layer = [kv_heads] * n_layers
        interval = _meta(meta, arch, "full_attention_interval")
        if isinstance(interval, int) and interval > 1:      # a hybrid: only every interval-th layer caches K/V
            per_layer = [kv_heads if (i + 1) % interval == 0 else 0 for i in range(n_layers)]
        elif any(k.startswith(f"{arch}.ssm.") for k in meta):
            notes.append("recurrent layers found without a per-layer table: the KV cache is counted for every layer (an upper bound)")
    tokens = [ctx] * n_layers
    window = _scalar(_meta(meta, arch, "attention.sliding_window"))
    if window:
        flags = swa_flags(meta, arch, n_layers)
        if flags is None:
            notes.append("sliding-window attention without a layer pattern in the header: the KV cache is counted at the full "
                         "context on every layer (an upper bound - llama.cpp keeps only the window on those layers); "
                         "--kv-gib plans with the real size")
        else:
            swa = min(ctx, -(-(window * parallel + SWA_UBATCH) // SWA_PAD) * SWA_PAD)
            tokens = [swa if f else ctx for f in flags]
            notes.append(f"sliding-window layers ({sum(flags)} of {n_layers}) are counted at {swa} tokens: llama.cpp's default "
                         "window cache (--swa-full would size them at the full context)")
    if any(k.startswith(f"{arch}.attention.kv_lora_rank") for k in meta):
        notes.append("multi-head latent attention: recent llama.cpp builds cache a compressed latent, smaller than counted "
                     "here (an upper bound)")
    return [h * (key_len + val_len) * t * KV_BYTES[kv] for h, t in zip(per_layer, tokens)], notes


def fit_from_end(costs: list, budget: float) -> int:
    """How many layers, counted from the last, fit in `budget` (llama.cpp offloads the last -ngl layers)."""
    n = 0
    for c in reversed(costs):
        if c > budget:
            break
        budget -= c
        n += 1
    return n


def override_tensor(n_cpu: int, n_layers: int) -> str:
    """The -ot pattern that keeps the experts of the first n_cpu layers in RAM (builds without --n-cpu-moe)."""
    if n_cpu >= n_layers:
        return r"\.ffn_.*_exps\.=CPU"
    return r"blk\.(" + "|".join(str(i) for i in range(n_cpu)) + r")\.ffn_.*_exps\.=CPU"


def plan_model(label: str, files: list, declared: int, absent: list, hw: dict, o) -> Plan:
    """The Plan for one model: `files` are the shards that were read, `hw` = {"vram": GiB, "ram": GiB}."""
    plan = Plan(target=label)
    meta = files[0].metadata
    arch = meta.get("general.architecture")
    plan.architecture = arch if isinstance(arch, str) else None
    if plan.architecture is None:
        plan.problems.append("no general.architecture: this is not the first shard of a model GGUF")
        return plan
    if absent or len(files) != declared:
        plan.problems.append(f"{len(files)} of {declared} shards found" + (": " + ", ".join(absent) if absent else "")
                             + " - the plan needs every shard's tensor table")
        return plan
    if arch == M.ARCH:
        plan.notes.append("this is the Flash-Next layout: Strata's own engine runs it, much faster (tools/model_check.py)")

    tensors = {t.name: t for f in files for t in f.tensors}
    unknown = sorted({t.type_name for t in tensors.values() if t.expected_bytes() is None})
    if unknown:
        plan.problems.append(f"tensor type(s) {', '.join(unknown)}: their size is not known to this tool")
        return plan
    n_layers = _meta(meta, arch, "block_count")
    if not isinstance(n_layers, int):
        n_layers = 1 + max((int(m[1]) for n in tensors if (m := BLOCK.match(n))), default=-1)
    if n_layers < 1:
        plan.problems.append("no layers found (no blk.N.* tensors)")
        return plan

    dense, experts, outputs, embed, other = [0] * n_layers, [0] * n_layers, 0, 0, 0
    for name, t in tensors.items():
        size, m = t.expected_bytes(), BLOCK.match(name)
        if name in OUTPUT:
            outputs += size
        elif name == EMBED:
            embed += size
        elif not m or int(m[1]) >= n_layers:
            other += size                                   # rope tables, extra (draft) layers: kept in RAM
        elif EXPERT.match(name):
            experts[int(m[1])] += size
        else:
            dense[int(m[1])] += size
    # tied embeddings (no output.weight): the output layer reads token_embd. On the GPU that is a second copy; on the CPU
    # llama.cpp points it at the same mmap pages, so only the GPU is charged for it
    tied = embed if EMBED in tensors and "output.weight" not in tensors else 0
    n_expert, used = _meta(meta, arch, "expert_count") or 0, _meta(meta, arch, "expert_used_count") or 0
    is_moe = sum(experts) > 0

    try:
        if o.kv_gib is not None:
            kv = [o.kv_gib * GIB / n_layers] * n_layers
        else:
            kv, notes = kv_layers(meta, arch, n_layers, o.ctx, o.kv, o.parallel)
            plan.notes += notes
    except ValueError as e:
        plan.problems.append(str(e))
        return plan

    # the placement: dense layers + KV from the last layer backwards, then (MoE) experts from the last layer backwards
    vram, ram = hw["vram"] * GIB, hw["ram"] * GIB
    budget = vram - o.reserve_vram * GIB
    costs = [dense[i] + kv[i] for i in range(n_layers)]
    on_gpu = fit_from_end(costs, budget - outputs - tied)
    if on_gpu == n_layers:
        ngl, out_on_gpu = ALL_LAYERS, True
        leftover = budget - outputs - tied - sum(costs)
    else:
        on_gpu = fit_from_end(costs, budget)                # the output layer stays on the CPU, and no expert is left room
        ngl, out_on_gpu, leftover = on_gpu, False, 0
    exp_on_gpu = 0
    if is_moe:
        for i in reversed(range(n_layers - on_gpu, n_layers)):
            if experts[i] > leftover:
                break
            leftover -= experts[i]
            exp_on_gpu += 1
    gpu_layers, gpu_exp = range(n_layers - on_gpu, n_layers), range(n_layers - exp_on_gpu, n_layers)
    gpu_bytes = (outputs + tied if out_on_gpu else 0) + sum(costs[i] for i in gpu_layers) + sum(experts[i] for i in gpu_exp)
    cpu_dense = sum(dense[i] for i in range(n_layers) if i not in gpu_layers)
    cpu_kv = sum(kv[i] for i in range(n_layers) if i not in gpu_layers)
    cpu_exp = sum(experts[i] for i in range(n_layers) if i not in gpu_exp)
    cpu_bytes = embed + other + (0 if out_on_gpu else outputs) + cpu_dense + cpu_kv + cpu_exp
    weights = sum(dense) + sum(experts) + outputs + embed + other

    plan.ngl = ngl
    plan.n_cpu_moe = (n_layers - exp_on_gpu) if is_moe else None
    plan.flags = ["-c", str(o.ctx), "-ngl", str(ngl)]
    if plan.n_cpu_moe:
        plan.flags_ot = plan.flags + ["-ot", override_tensor(plan.n_cpu_moe, n_layers)]
        plan.flags += ["--n-cpu-moe", str(plan.n_cpu_moe)]
    else:
        plan.flags_ot = list(plan.flags)
    tail = (["-ctk", o.kv, "-ctv", o.kv] if o.kv != "f16" else []) + (["-t", str(o.threads)] if o.threads else []) + \
        ["-np", str(o.parallel), "--jinja", "--host", o.host, "--port", str(o.port)] + \
        (["--alias", o.alias] if o.alias else [])
    plan.flags += tail
    plan.flags_ot += tail

    plan.facts = {
        "layers": n_layers, "experts": n_expert or None, "experts_used": used or None, "moe": is_moe,
        "weights_gib": round(weights / GIB, 1), "dense_gib": round((weights - sum(experts)) / GIB, 1),
        "experts_gib": round(sum(experts) / GIB, 1),
        "kv_gib": round(sum(kv) / GIB, 1), "layers_on_gpu": on_gpu, "output_on_gpu": out_on_gpu,
        "expert_layers_on_gpu": exp_on_gpu if is_moe else None,
        "gpu_gib": round(gpu_bytes / GIB, 1), "cpu_gib": round(cpu_bytes / GIB, 1),
        "vram_gib": round(hw["vram"], 1), "ram_gib": round(hw["ram"], 1), "ctx": o.ctx, "kv_type": o.kv,
    }

    # the verdict: does what stays in RAM fit beside the system?
    if cpu_bytes + o.reserve_ram * GIB <= ram:
        plan.status = "ok"
    elif cpu_bytes <= ram:
        plan.status = "tight"
        plan.notes.append(f"RAM is tight: {cpu_bytes / GIB:.1f} GiB stay in RAM of {hw['ram']:.1f}, and {o.reserve_ram:g} GiB "
                          "are set aside for the system (--reserve-ram) - close other programs, or lower --ctx")
    else:
        plan.status = "slow"
        plan.notes.append(f"{cpu_bytes / GIB:.1f} GiB would stay in RAM of {hw['ram']:.1f}: the rest is read from the SSD "
                          "(mmap) while it answers - far slower. A smaller quantization or a shorter --ctx would fit")
    if on_gpu < n_layers:
        plan.notes.append(f"the dense layers do not all fit {hw['vram']:.1f} GiB of VRAM: {n_layers - on_gpu} of {n_layers} "
                          "run on the CPU (slow); lower --ctx, use --kv q8_0, or a smaller quantization")
    if o.kv != "f16":
        plan.notes.append("a quantized V cache needs flash attention: add `-fa on` (current llama.cpp) or `-fa` (older builds) "
                          "to the flags; --run adds the right one for the build it finds")
    elif is_moe and plan.n_cpu_moe:
        plan.notes.append("--kv q8_0 halves the KV cache (more experts on the GPU); it needs llama.cpp's flash attention")

    # a roofline: the bytes the CPU reads per token, against the RAM bandwidth the user gave
    if o.ram_bw and plan.status in ("ok", "tight"):
        per_token = sum(dense[i] for i in range(n_layers) if i not in gpu_layers)
        if is_moe and n_expert and used:
            per_token += sum(experts[i] for i in range(n_layers) if i not in gpu_exp) * used / n_expert
        if per_token > 0:
            plan.ceiling_tok_s = round(o.ram_bw * 1e9 / per_token, 1)
    return plan


# ------------------------------------------------------------------------------------------------ llama-server
def find_server(explicit: str | None) -> str:
    """The llama-server to run: --llama-server, else $LLAMA_SERVER, else PATH. A path the user named is never replaced
    by another binary: if it is not an executable file, that is the error."""
    named = explicit or os.environ.get("LLAMA_SERVER")
    if named:
        found = named if os.path.isfile(named) and os.access(named, os.X_OK) else shutil.which(named)
        if not found:
            raise FileNotFoundError(f"{named}: not an executable file (from {'--llama-server' if explicit else '$LLAMA_SERVER'})")
        return found
    for n in SERVER_NAMES:
        if shutil.which(n):
            return shutil.which(n)
    raise FileNotFoundError("llama-server not found: build or download llama.cpp, then pass --llama-server <path> "
                            "(or set LLAMA_SERVER, or put it on PATH)")


def probe_help(binary: str) -> str:
    """`llama-server --help`'s text (stdout and stderr), '' when it cannot be read."""
    try:
        r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60)
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("could not read %s --help (%s): assuming a current build", binary, e)
        return ""


def build_command(binary: str, model: str, plan: Plan, help_text: str) -> list:
    """llama-server's command line for this plan, spelled for the build whose --help is `help_text` ('' = current)."""
    cmd = [binary, "-m", model] + (plan.flags if not help_text or "--n-cpu-moe" in help_text else plan.flags_ot)
    if plan.facts.get("kv_type") != "f16" and help_text:        # a quantized V cache needs flash attention
        if "on|off|auto" in help_text:
            cmd += ["-fa", "on"]
        elif "--flash-attn" in help_text:
            cmd += ["-fa"]
    return cmd


def port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host if host not in ("0.0.0.0", "") else "127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def run_server(cmd: list) -> int:
    """Runs llama-server until it exits. Ctrl-C, SIGTERM and SIGHUP (a service manager's stop, a closed terminal) stop it
    too: the child is terminated, then killed after 15 s, never left running without its parent."""
    log.info("starting: %s", join(cmd))
    try:
        proc = subprocess.Popen(cmd)
    except OSError as e:
        print(f"\nnot starting: {cmd[0]}: {e.strerror or e}", file=sys.stderr)
        return 2

    def stop(signum, frame):
        raise KeyboardInterrupt

    old = {}
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            try:
                old[getattr(signal, name)] = signal.signal(getattr(signal, name), stop)
            except ValueError:                              # not the main thread: the caller's handlers stay
                pass
    try:
        return proc.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)


def join(cmd: list) -> str:
    """The command line as a shell takes it (cmd.exe quoting on Windows: its | & ( ) are not safe bare)."""
    if sys.platform != "win32":
        return shlex.join(cmd)
    return " ".join(f'"{c}"' if any(ch in c for ch in "|&()<>^") and '"' not in c else subprocess.list2cmdline([c])
                    for c in cmd)


# ------------------------------------------------------------------------------------------------ output
VERDICT = {"ok": "OK     - fits: what stays in RAM leaves room for the system", "tight": "TIGHT  - fits, with little RAM to spare",
           "slow": "SLOW   - does not fit RAM: the SSD serves the rest", "no": "NO     - cannot be planned",
           "error": "ERROR  - could not read it"}


def show(p: Plan) -> None:
    print(f"\n{p.target}")
    print(f"  verdict      {VERDICT[p.status]}")
    f = p.facts
    if p.architecture:
        print(f"  architecture {p.architecture}")
    if f:
        kind = "dense, " if not f["moe"] else \
            f"{f['experts']} experts ({f['experts_used']} used), " if f["experts"] else "a mixture of experts, "
        print(f"  model        {f['layers']} layers, {kind}{f['weights_gib']} GiB of weights "
              f"(dense {f['dense_gib']} + experts {f['experts_gib']})")
        print(f"  this PC      {f['vram_gib']} GiB VRAM, {f['ram_gib']} GiB RAM; context {f['ctx']}, KV {f['kv_type']} "
              f"{f['kv_gib']} GiB")
        gpu = f"{f['layers_on_gpu']} of {f['layers']} layers" + (" + output" if f["output_on_gpu"] else "")
        if f["moe"]:
            gpu += f"; experts of the last {f['expert_layers_on_gpu']} layers"
        print(f"  on the GPU   {gpu} (~{f['gpu_gib']} GiB)")
        print(f"  in RAM       the rest (~{f['cpu_gib']} GiB)")
        if p.ceiling_tok_s:
            print(f"  ceiling      ~{p.ceiling_tok_s} tokens/s from the RAM bandwidth given (the real speed is lower)")
    for label, items in (("problem", p.problems), ("note", p.notes)):
        for i, line in enumerate(items):
            print(f"  {label if i == 0 else '':<12} {line}")
    if p.flags:
        print(f"  flags        {join(p.flags)}")
        if p.n_cpu_moe:
            print(f"  older llama  {join(p.flags_ot)}   (builds without --n-cpu-moe)")
        print("  run          llama-server -m <the file> <flags>   (or: --run)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plan llama.cpp's expert offload for a GGUF on this PC (header only).",
                                 epilog="A target is a .gguf file (the first shard), a URL of one, or a Hugging Face "
                                        "owner/repo; --run needs a local file.")
    M.add_source_options(ap)
    ap.add_argument("--vram", type=float, help="GiB of VRAM to plan for (default: the first NVIDIA card's free VRAM)")
    ap.add_argument("--ram", type=float, help="GiB of RAM (default: installed)")
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX, help=f"context in tokens (default {DEFAULT_CTX})")
    ap.add_argument("--kv", choices=sorted(KV_BYTES), default="f16", help="KV cache type (default f16)")
    ap.add_argument("--kv-gib", type=float, help="the KV cache's size in GiB, for a model whose header does not tell")
    ap.add_argument("--reserve-vram", type=float, default=1.5, help="GiB of VRAM left to CUDA and compute buffers (1.5)")
    ap.add_argument("--reserve-ram", type=float, default=8.0, help="GiB of RAM left to the system and other programs (8)")
    ap.add_argument("--ram-bw", type=float, help="RAM bandwidth in GB/s: adds a decode-speed ceiling to the plan")
    ap.add_argument("--threads", type=int, help="llama-server -t (default: its own choice)")
    ap.add_argument("--parallel", type=int, default=1, help="slots, llama-server -np (default 1: one request at a time, "
                                                            "like Strata; --ctx is shared between them)")
    ap.add_argument("--host", default="127.0.0.1", help="address to serve on (default 127.0.0.1, like Strata)")
    ap.add_argument("--port", type=int, default=8080, help="port (default 8080, like Strata)")
    ap.add_argument("--alias", help="the model name clients see")
    ap.add_argument("--json", action="store_true", help="print the plans as JSON")
    ap.add_argument("--run", action="store_true", help="start llama-server with the plan (one local file)")
    ap.add_argument("--llama-server", help="path of llama-server (default $LLAMA_SERVER, then PATH)")
    ap.add_argument("--force", action="store_true", help="--run even when the plan is SLOW")
    a = ap.parse_args(argv)
    src = M.source_from(a, ap)
    for name in ("ctx", "port", "parallel"):
        if getattr(a, name) < 1:
            ap.error(f"--{name} must be positive")
    for name in ("vram", "ram", "kv_gib", "ram_bw"):
        v = getattr(a, name)
        if v is not None and v <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if a.reserve_vram < 0 or a.reserve_ram < 0 or (a.threads is not None and a.threads < 1):
        ap.error("--reserve-vram / --reserve-ram cannot be negative, --threads must be positive")
    if a.run and (len(a.targets) != 1 or not os.path.isfile(a.targets[0])):
        ap.error("--run needs exactly one target, a local .gguf file (download it first)")
    hw = {"vram": a.vram if a.vram is not None else detect_vram_gib(),
          "ram": a.ram if a.ram is not None else detect_ram_gib()}
    for key in ("vram", "ram"):
        if hw[key] is None:
            ap.error(f"could not detect this PC's {key.upper()}: pass --{key} <GiB>")
    log.info("planning for %.1f GiB VRAM, %.1f GiB RAM", hw["vram"], hw["ram"])

    plans = []
    for arg in a.targets:
        try:
            for label, first in M.resolve(arg, a, src):
                try:
                    files, names, absent = M.read_headers(first, src)
                    plan = plan_model(label, files, len(names), absent, hw, a)
                    if not first.startswith(("http://", "https://")):
                        # -m is the first shard under the name it was given (llama.cpp finds the others beside it by name):
                        # absolute, but a symlink - the Hugging Face cache's - is not followed to its hash-named blob
                        plan.model_path = os.path.abspath(os.path.join(os.path.dirname(first), names[0]))
                    plans.append(plan)
                except M.READ_ERRORS as e:
                    log.debug("%s", label, exc_info=True)
                    plans.append(Plan(target=label, status="error", problems=[M.why(e)]))
        except M.READ_ERRORS as e:
            log.debug("%s", arg, exc_info=True)
            plans.append(Plan(target=arg, status="error", problems=[M.why(e)]))
    if a.json:
        print(json.dumps([dataclasses.asdict(p) for p in plans], indent=2))
    else:
        for p in plans:
            show(p)
        if len(plans) > 1:
            print("\nsummary")
            for p in plans:
                print(f"  {p.status.upper():<6} {p.target}")

    if a.run:
        p = plans[0]
        if p.status in ("no", "error") or (p.status == "slow" and not a.force):
            print(f"\nnot starting: the plan is {p.status.upper()}" + (" (--force starts it anyway)" if p.status == "slow" else ""),
                  file=sys.stderr)
            return 1
        if port_in_use(a.host, a.port):
            print(f"\nnot starting: {a.host}:{a.port} is in use (Strata or another server is running)", file=sys.stderr)
            return 1
        try:
            binary = find_server(a.llama_server)
        except FileNotFoundError as e:
            print(f"\nnot starting: {e}", file=sys.stderr)
            return 2
        cmd = build_command(binary, p.model_path, p, probe_help(binary))
        print(f"\nstarting: {join(cmd)}\nAPI: http://{a.host if a.host != '0.0.0.0' else '127.0.0.1'}:{a.port}/v1", flush=True)
        return run_server(cmd)
    if any(p.status == "error" for p in plans):
        return 2
    return 1 if any(p.status == "no" for p in plans) else 0


if __name__ == "__main__":
    raise SystemExit(main())
