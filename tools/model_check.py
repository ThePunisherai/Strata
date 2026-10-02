"""tools/model_check.py - can Strata run this model file?  Reads the GGUF header only: nothing big is downloaded.

    python tools/model_check.py /models/Model-IQ3_XXS-00001-of-00002.gguf
    python tools/model_check.py https://huggingface.co/<owner>/<repo>/resolve/main/<file>-00001-of-0000N.gguf
    python tools/model_check.py <owner>/<repo> --match IQ3          (every GGUF of a Hugging Face repo)
    python tools/model_check.py https://huggingface.co/<owner>/<repo> <owner>/<repo2> --json

Strata is specialised to ONE architecture, `qwen4exp` (the Qwen3.8-Flash-Next layout: 48 layers, width 2560,
512 experts of which 10 are used - 256 in the Coder).  The engine refuses every other `general.architecture`
(include/strata/artifact/gguf_reader.hpp: check_architecture), and its CUDA/CPU kernels are written for that
geometry, so another architecture is a new model implementation, not a setting.  A fine-tune, an abliteration or
an "uncensored" release of Flash-Next keeps the layout and runs; this tool says which side a file is on, from the
header alone (a local file, or a URL / Hugging Face repo through HTTP Range requests of a few MiB per shard).

A split model ("-00001-of-0000N.gguf") is read shard by shard: the first shard has the metadata, the layers are
spread over all of them.  Verdicts: ok = the engine's own checks pass and every expert type is one it reads;
check = it should run, with a condition named below (an experimental route, a rounding step); no = it will not.

Exit status: 0 every file ok/check, 1 at least one `no`, 2 a file could not be read.  `--json` prints the reports;
-v / -vv log what is read to stderr.  HF_TOKEN reaches a gated repo (sent to the Hub only, never to a redirect).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

log = logging.getLogger("model_check")

ARCH = "qwen4exp"
# The metadata the engine compares (check_architecture, with the canonical values) and the kernels are built for.
GEOMETRY = {"block_count": 48, "embedding_length": 2560, "attention.head_count": 24, "attention.head_count_kv": 2}
EXPERTS_USED = 10
N_FF = 640                                                    # the routed experts' width (kernels: FF = 640)
EXPERT_COUNTS = {512: "the original model", 256: "the Coder (half the experts)"}
ROLES = ("gate", "up", "down")

# Expert tensor types the engine reads: None = as the installer's models, text = an experimental route (condition).
KQUANT = "K-quant experts: the experimental UD-Q4_K_XL / Q4_K_S route (docs/UNSLOTH_Q4.md, docs/ORCA_Q4_K_S.md)"
GATE_UP = {"Q2_0": None, "IQ1_M": None, "IQ2_XXS": None, "IQ2_XS": None, "IQ2_S": None, "IQ3_XXS": None,
           "IQ3_S": None, "Q4_K": KQUANT, "Q5_K": KQUANT}
DOWN = {"Q2_0": None, "IQ4_NL": None, "Q5_1": KQUANT, "Q8_0": KQUANT}
DOWN_REFUSED = {"Q5_0": "the GPU expert kernels have no Q5_0 down matrix: the engine refuses to start on it (docs/ORCA.md)"}

# What a Flash-Next file has that no other architecture does (all layers: routed experts and their router).
MODEL_TENSORS = ("token_embd.weight", "per_layer_token_embd.weight", "output_hc_down.weight")
LAYER_TENSORS = tuple(f"ffn_{r}_exps.weight" for r in ROLES) + ("ffn_gate_inp.weight",)

SPLIT = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<n>\d{5})\.gguf$")
HF_HOSTS = ("huggingface.co", "hf.co")


def is_hub(netloc: str) -> bool:
    """huggingface.co, hf.co or one of their subdomains - not a host that merely ends the same way."""
    host = netloc.rsplit("@", 1)[-1].split(":")[0].lower()
    return any(host == h or host.endswith("." + h) for h in HF_HOSTS)


CHUNK = 8 << 20                                               # one Range request: most headers fit in one or two
MAX_HEADER = 256 << 20                                        # a header larger than this is not a GGUF header
UA = "strata-model-check"
GIB = 1 << 30


# ------------------------------------------------------------------------------------------------ the header
class RangeReader:
    """A read-only file over an HTTP URL: `read` / `tell` for GGUFFile._parse, fetched in CHUNK-byte Range windows."""

    def __init__(self, url: str, opener, headers: dict, timeout: float):
        self.url, self.opener, self.headers, self.timeout = url, opener, headers, timeout
        self.pos, self.start, self.buf, self.requests = 0, 0, b"", 0

    def tell(self) -> int:
        return self.pos

    def _fill(self) -> None:
        req = urllib.request.Request(self.url, headers={
            **self.headers, "Range": f"bytes={self.pos}-{self.pos + CHUNK - 1}", "User-Agent": UA})
        self.requests += 1
        log.debug("GET %s Range: bytes=%d-%d", self.url, self.pos, self.pos + CHUNK - 1)
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                if r.status == 200 and self.pos:
                    raise OSError("the server ignores Range requests (HTTP 200 for a range)")
                data = r.read(CHUNK)                          # a 200 from a server that ignores Range stops here too
        except urllib.error.HTTPError as e:
            if e.code != 416:                                 # 416: the position is past the end of the file
                raise
            data = b""
        self.start, self.buf = self.pos, data

    def read(self, n: int) -> bytes:
        if self.pos + n > MAX_HEADER:
            raise OSError(f"the header reaches past {MAX_HEADER >> 20} MiB: not a GGUF header")
        out = bytearray()
        while len(out) < n:
            off = self.pos - self.start
            if not 0 <= off < len(self.buf):
                self._fill()
                off = 0
                if not self.buf:
                    break
            take = self.buf[off:off + n - len(out)]
            out += take
            self.pos += len(take)
        return bytes(out)


class RemoteGGUF(G.GGUFFile):
    """tools/gguf_reader.GGUFFile over a URL: the same parser, fed by a RangeReader."""

    def __init__(self, url: str, opener, headers: dict, timeout: float):
        self.path = pathlib.PurePosixPath(urllib.parse.unquote(urllib.parse.urlparse(url).path))
        self.metadata, self.tensors, self.version, self.alignment = {}, [], 0, 32
        reader = RangeReader(url, opener, headers, timeout)
        self._parse(reader)
        self.data_start = self._data_start
        log.info("%s: header read in %d request(s), %d tensors", self.path.name, reader.requests, len(self.tensors))


class _KeepAuthOnHost(urllib.request.HTTPRedirectHandler):
    """Follows redirects (the Hub answers with a CDN address) but sends the token to the original host only."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
        return new


def shard_names(name: str):
    """['m-00001-of-00003.gguf', ...] for a split model's first shard (any shard's name), else [name]."""
    m = SPLIT.match(name)
    if not m:
        return [name]
    return [f"{m['stem']}-{i:05d}-of-{m['n']}.gguf" for i in range(1, int(m["n"]) + 1)]


# ------------------------------------------------------------------------------------------------ the verdict
@dataclasses.dataclass
class Report:
    target: str
    status: str = "no"                      # ok | check | no | error
    architecture: str | None = None
    shards: int = 1                         # declared by the file name
    shards_read: int = 0
    facts: dict = dataclasses.field(default_factory=dict)
    problems: list = dataclasses.field(default_factory=list)
    warnings: list = dataclasses.field(default_factory=list)
    next: list = dataclasses.field(default_factory=list)


def _u(meta: dict, key: str):
    v = meta.get(key)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _bf16_wanted():
    """iq_pack's table of the tensors the engine reads as BF16 (needs numpy, which the setup's .venv has)."""
    try:
        import iq_pack
        return iq_pack.FORM, iq_pack.NATIVE_PLE_KEY
    except ImportError as e:
        log.info("packing hints skipped: %s (use the setup's .venv python)", e)
        return None, set()


def analyse(target: str, files: list, declared: int) -> Report:
    """The Report for one model: `files` are the shards that could be read (the first has the metadata)."""
    rep = Report(target=target, shards=declared, shards_read=len(files))
    meta = files[0].metadata
    arch = meta.get("general.architecture")
    rep.architecture = arch if isinstance(arch, str) else None
    if rep.architecture is None:
        rep.problems.append("no general.architecture: this is not the first shard of a model GGUF")
        return rep
    if arch != ARCH:
        for k in ("block_count", "expert_count", "embedding_length"):
            if _u(meta, f"{arch}.{k}") is not None:
                rep.facts[k] = _u(meta, f"{arch}.{k}")
        rep.problems.append(
            f"architecture is '{arch}', Strata runs '{ARCH}' only (the engine refuses anything else, and its kernels are "
            f"written for that one layout). Supporting '{arch}' would be a new model implementation - loader and "
            "kernels - not a configuration; tools/llama_plan.py plans llama.cpp's expert offload for this file on your PC")
        return rep

    # 1. the metadata the engine checks itself
    for key, want in GEOMETRY.items():
        got = _u(meta, f"{ARCH}.{key}")
        rep.facts[key] = got
        if got is None:
            rep.problems.append(f"missing {ARCH}.{key}")
        elif got != want:
            rep.problems.append(f"{ARCH}.{key} = {got}, the kernels are built for {want}")
    n_exp, used = _u(meta, f"{ARCH}.expert_count"), _u(meta, f"{ARCH}.expert_used_count")
    rep.facts.update(expert_count=n_exp, expert_used_count=used)
    if n_exp is None or used is None:
        rep.problems.append(f"missing {ARCH}.expert_count / {ARCH}.expert_used_count")
    else:
        if used != EXPERTS_USED:
            rep.problems.append(f"{used} experts are used per token, the kernels are built for {EXPERTS_USED}")
        if n_exp in EXPERT_COUNTS:
            rep.facts["variant"] = EXPERT_COUNTS[n_exp]
        else:
            rep.warnings.append(f"{n_exp} experts per layer: only 512 (original) and 256 (Coder) are known to run")

    # 2. the tensors, over every shard that was read
    tensors, dup = {}, []
    for f in files:
        for t in f.tensors:
            if t.name in tensors:
                dup.append(t.name)
            tensors[t.name] = t
    if dup:
        rep.problems.append(f"the same tensor in two shards: {', '.join(dup[:3])}")
    complete = len(files) == declared
    layers = rep.facts.get("block_count") or GEOMETRY["block_count"]
    if complete:
        want = list(MODEL_TENSORS) + [f"blk.{i}.{n}" for i in range(layers) for n in LAYER_TENSORS]
        missing = [n for n in want if n not in tensors]
        if missing:
            rep.problems.append(f"{len(missing)} tensors of the {ARCH} layout are missing, e.g. {', '.join(missing[:3])}")
    else:
        rep.warnings.append(f"{len(files)} of {declared} shards read: the tensors of the others are not checked")

    # 3. the experts: shape, one type per role, a type the kernels read
    gu_types, d_types, expert_bytes = set(), set(), 0
    for name, t in tensors.items():
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if not m:
            continue
        role = m.group(2)
        expert_bytes += t.expected_bytes() or 0
        (d_types if role == "down" else gu_types).add(t.type_name)
        shape_ok = len(t.shape) == 3 and sorted(t.shape[:2]) == sorted([N_FF, GEOMETRY["embedding_length"]]) and \
            (n_exp is None or t.shape[2] == n_exp)
        if not shape_ok:
            rep.problems.append(f"{name} has shape {t.shape}, expected [{GEOMETRY['embedding_length']}, {N_FF}, "
                                f"{n_exp}] (down: transposed)")
            break
    for i in range(layers if complete else 0):
        g, u = tensors.get(f"blk.{i}.ffn_gate_exps.weight"), tensors.get(f"blk.{i}.ffn_up_exps.weight")
        if g is not None and u is not None and g.type_name != u.type_name:
            rep.problems.append(f"layer {i}: gate ({g.type_name}) and up ({u.type_name}) differ in type")
            break
    rep.facts["expert_types"] = {"gate_up": sorted(gu_types), "down": sorted(d_types)}
    rep.facts["experts_gib"] = round(expert_bytes / GIB, 1)
    conditions: dict = {}
    for role, types, ok in (("gate/up", gu_types, GATE_UP), ("down", d_types, DOWN)):
        for ty in sorted(types):
            if ty in DOWN_REFUSED and role == "down":
                rep.problems.append(f"{role} experts are {ty}: {DOWN_REFUSED[ty]}")
            elif ty not in ok:
                rep.problems.append(f"{role} experts are {ty}, not a type Strata's expert kernels read "
                                    f"({', '.join(sorted(ok))})")
            elif ok[ty]:
                conditions.setdefault(ok[ty], []).append(f"{role} {ty}")
    for text, which in conditions.items():
        rep.warnings.append(f"{', '.join(which)} - {text}")
    if not expert_bytes and complete and not rep.problems:
        rep.problems.append("no routed-expert tensors found")

    # 4. the small projections the engine reads as BF16: converted (rounded) by iq_pack --compat-bf16
    form, native_key = _bf16_wanted()
    if form:
        stray = [n for n, t in tensors.items()
                 if form.get(re.sub(r"^blk\.\d+\.", "", n)) == "BF16" and t.type_name != "BF16"
                 and not (n == "blk.1.ple_key.weight" and t.type_name in native_key)]
        rep.facts["bf16_conversions"] = len(stray)
        if stray:
            kinds = sorted({tensors[n].type_name for n in stray})
            rep.next.append(f"{len(stray)} small projections are stored as {'/'.join(kinds)} where the engine reads BF16: "
                            "pack with --compat-bf16 (rounds them to BF16; the original precision is not restored)")

    # 5. the verdict and the route
    rep.status = "no" if rep.problems else "check" if rep.warnings else "ok"
    if rep.status != "no":
        stem = SPLIT.sub(r"\g<stem>", files[0].path.name) if SPLIT.match(files[0].path.name) \
            else re.sub(r"\.gguf$", "", files[0].path.name)
        rep.next += [
            "setup's menu installs the qwen / swift / coder / unsloth families; any other file is packed by hand "
            "(docs/ORCA.md):",
            f"python tools/iq_pack.py --gguf <first shard> --out packs/{stem.lower()}"
            + (" --compat-bf16" if rep.facts.get("bf16_conversions") else ""),
            "keep this model's own tokenizer (the pack exports it); the draft head is the original model's, so its "
            "acceptance - and the speed - must be measured for this fine-tune",
        ]
    return rep


# ------------------------------------------------------------------------------------------------ targets
class Source:
    """Where the headers come from: files on disk, or URLs (a Hugging Face endpoint, with an optional token)."""

    def __init__(self, token: str | None, endpoint: str, timeout: float):
        self.token, self.endpoint, self.timeout = token, endpoint.rstrip("/"), timeout
        self.opener = urllib.request.build_opener(_KeepAuthOnHost)

    def _headers(self, url: str) -> dict:
        host = urllib.parse.urlparse(url).netloc
        ours = host == urllib.parse.urlparse(self.endpoint).netloc or is_hub(host)
        return {"Authorization": f"Bearer {self.token}"} if self.token and ours else {}

    def open_url(self, url: str) -> RemoteGGUF:
        return RemoteGGUF(url, self.opener, self._headers(url), self.timeout)

    def repo_files(self, repo: str, rev: str) -> list:
        """The first-shard URLs of the repo's model GGUFs (no mmproj, no later shards)."""
        api = f"{self.endpoint}/api/models/{repo}" + ("" if rev == "main" else f"/revision/{urllib.parse.quote(rev, safe='')}")
        req = urllib.request.Request(api, headers={**self._headers(api), "User-Agent": UA})
        log.info("listing %s", api)
        with self.opener.open(req, timeout=self.timeout) as r:
            siblings = json.load(r).get("siblings", [])
        names = []
        for s in siblings:
            n = s.get("rfilename", "")
            base = n.rsplit("/", 1)[-1]
            m = SPLIT.match(base)
            if n.endswith(".gguf") and "mmproj" not in base.lower() and (not m or m["no"] == "00001"):
                names.append(n)
        return [f"{self.endpoint}/{repo}/resolve/{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(n)}"
                for n in sorted(names)]


def classify(arg: str, default_rev: str, endpoint: str):
    """('file', path) | ('url', url) | ('repo', (owner/repo, rev)) for a command-line target."""
    if os.path.exists(arg):
        return "file", arg
    if arg.lower().endswith(".gguf") and not urllib.parse.urlparse(arg).scheme:
        raise FileNotFoundError(f"{arg}: no such file")
    u = urllib.parse.urlparse(arg)
    if u.scheme in ("http", "https"):
        seg = [urllib.parse.unquote(s) for s in u.path.strip("/").split("/")]
        if is_hub(u.netloc) and len(seg) >= 2:
            if len(seg) >= 5 and seg[2] in ("resolve", "blob") and seg[-1].endswith(".gguf"):
                return "url", f"{u.scheme}://{u.netloc}/{seg[0]}/{seg[1]}/resolve/" + \
                    "/".join(urllib.parse.quote(s) for s in seg[3:])
            return "repo", (f"{seg[0]}/{seg[1]}", seg[3] if len(seg) > 3 and seg[2] == "tree" else default_rev)
        return "url", arg
    if re.fullmatch(r"[\w.-]+/[\w.-]+", arg):
        return "repo", (arg, default_rev)
    raise ValueError(f"{arg}: not a file, a URL or an owner/repo")


def read_headers(first: str, src: Source):
    """(files, names, absent): the headers of every shard of the model whose first shard is `first` (a path or a URL),
    the shard names its file name declares, and those of them that could not be found."""
    is_url = first.startswith(("http://", "https://"))
    base, _, name = first.rpartition("/") if is_url else (str(pathlib.Path(first).parent), "", pathlib.Path(first).name)
    names = shard_names(urllib.parse.unquote(name) if is_url else name)
    files, absent = [], []
    for n in names:
        where = f"{base}/{urllib.parse.quote(n)}" if is_url else str(pathlib.Path(base) / n)
        try:
            files.append(src.open_url(where) if is_url else G.GGUFFile(pathlib.Path(where)))
        except FileNotFoundError:
            absent.append(n)
        except urllib.error.HTTPError as e:
            if e.code == 404 and files:
                absent.append(n)
                continue
            raise
    if not files:
        raise FileNotFoundError(f"{names[0]}: not found")
    return files, names, absent


def read_model(label: str, first: str, src: Source) -> Report:
    """Reads every shard of the model whose first shard is `first` and judges it."""
    files, names, absent = read_headers(first, src)
    rep = analyse(label, files, len(names))
    if absent and len(files) != len(names):
        rep.warnings.append("shards not found: " + ", ".join(absent))
    return rep


def resolve(arg: str, a, src: Source) -> list:
    """[(label, first-shard path or URL)] for one command-line target; a repo expands to its model files."""
    kind, what = classify(arg, a.rev, src.endpoint)
    if kind != "repo":
        return [(arg, what)]
    repo, rev = what
    urls = src.repo_files(repo, rev)
    picked = [u for u in urls if a.match.lower() in urllib.parse.unquote(u).lower()]
    log.info("%s: %d model file(s), %d match", repo, len(urls), len(picked))
    if not picked:
        raise FileNotFoundError(f"{repo}: no model .gguf file" + (f" matching '{a.match}'" if a.match else ""))
    if len(picked) > a.max_files:
        log.warning("%s: %d more file(s) not read (--max-files %d); narrow with --match",
                    repo, len(picked) - a.max_files, a.max_files)
    return [(f"{repo} :: {urllib.parse.unquote(u).rsplit('/', 1)[-1]}", u) for u in picked[:a.max_files]]


def run_target(arg: str, a, src: Source) -> list:
    reports = []
    try:
        for label, first in resolve(arg, a, src):
            try:
                reports.append(read_model(label, first, src))
            except (OSError, struct.error, KeyError, ValueError, UnicodeError, MemoryError) as e:
                log.debug("%s", label, exc_info=True)
                reports.append(Report(target=label, status="error", problems=[why(e)]))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.debug("%s", arg, exc_info=True)
        reports.append(Report(target=arg, status="error", problems=[why(e)]))
    return reports


def why(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        hint = {401: " (private or gated: set HF_TOKEN)", 403: " (refused: a gated repo, or a network policy blocks the host)",
                404: " (not found)"}.get(e.code, "")
        return f"HTTP {e.code} from {urllib.parse.urlparse(e.url).netloc}{hint}"
    if isinstance(e, urllib.error.URLError):
        return f"cannot reach the host: {e.reason}" + (" (a proxy or network policy blocks it)" if "403" in str(e.reason) else "")
    return f"{type(e).__name__}: {e}" if not isinstance(e, FileNotFoundError) else str(e)


# ------------------------------------------------------------------------------------------------ output
VERDICT = {"ok": "OK     - Strata can run it", "check": "CHECK  - should run, with a condition",
           "no": "NO     - Strata cannot run it", "error": "ERROR  - could not read it"}


def show(r: Report) -> None:
    print(f"\n{r.target}")
    print(f"  verdict      {VERDICT[r.status]}")
    if r.architecture:
        print(f"  architecture {r.architecture}")
    f = r.facts
    if r.architecture == ARCH:
        print(f"  layout       {f.get('block_count')} layers, width {f.get('embedding_length')}, heads "
              f"{f.get('attention.head_count')}/{f.get('attention.head_count_kv')}, experts {f.get('expert_count')} "
              f"({f.get('expert_used_count')} used)" + (f" - {f['variant']}" if "variant" in f else ""))
        t = f.get("expert_types", {})
        print(f"  experts      gate/up {'+'.join(t.get('gate_up', [])) or '?'}, down {'+'.join(t.get('down', [])) or '?'}"
              f", {f.get('experts_gib')} GiB (RAM: about that plus 10 GB for the system, README)")
    elif f:
        print("  layout       " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in f.items()))
    if r.shards > 1:
        print(f"  shards       {r.shards_read} of {r.shards} read")
    for label, items in (("problem", r.problems), ("note", r.warnings), ("next", r.next)):
        for i, line in enumerate(items):
            print(f"  {label if i == 0 else '':<12} {line}")


def add_source_options(ap: argparse.ArgumentParser) -> None:
    """The targets and where their headers come from - shared with tools/llama_plan.py."""
    ap.add_argument("targets", nargs="+", help="file, URL or owner/repo")
    ap.add_argument("--rev", default="main", help="branch, tag or commit of a Hugging Face repo (default main)")
    ap.add_argument("--match", default="", help="of a repo's files, read those whose name contains this (e.g. IQ3_XXS)")
    ap.add_argument("--max-files", type=int, default=12, help="most files read per repo (default 12)")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    help="the Hub address (default $HF_ENDPOINT or https://huggingface.co)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                    help="Hub token for gated repos (default $HF_TOKEN)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per request (default 60)")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v: what is read; -vv: every request")


def source_from(a, ap: argparse.ArgumentParser) -> Source:
    """Logging and the Source from the options add_source_options defined (a usage error when they are out of range)."""
    logging.basicConfig(level=[logging.WARNING, logging.INFO, logging.DEBUG][min(a.verbose, 2)],
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if a.max_files < 1 or a.timeout <= 0:
        ap.error("--max-files and --timeout must be positive")
    return Source(a.token, a.endpoint, a.timeout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Can Strata run this model file? Reads the GGUF header only.",
                                 epilog="A target is a .gguf file (the first shard), a URL of one, or a Hugging Face "
                                        "owner/repo (or its page URL).")
    add_source_options(ap)
    ap.add_argument("--json", action="store_true", help="print the reports as JSON")
    a = ap.parse_args(argv)
    src = source_from(a, ap)
    reports = [r for t in a.targets for r in run_target(t, a, src)]
    if a.json:
        print(json.dumps([dataclasses.asdict(r) for r in reports], indent=2))
    else:
        for r in reports:
            show(r)
        if len(reports) > 1:
            print("\nsummary")
            for r in reports:
                print(f"  {r.status.upper():<6} {r.target}")
    if any(r.status == "error" for r in reports):
        return 2
    return 1 if any(r.status == "no" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
