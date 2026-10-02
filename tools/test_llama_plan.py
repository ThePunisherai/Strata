"""llama_plan's memory arithmetic and its llama-server command lines, over synthetic GGUF headers (no model, no GPU,
no llama.cpp: the server is a script that records its arguments).

    python -m unittest tools.test_llama_plan
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import socket
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import llama_plan as L  # noqa: E402
import model_check as M  # noqa: E402
import test_model_check as T  # noqa: E402

GIB = 1 << 30
LAYERS, EMB, HEADS, KV_HEADS, KEY, FF, N_EXP, USED, VOCAB = 48, 2048, 32, 4, 128, 768, 128, 8, 151936


def moe_meta(arch="qwen3moe", **over) -> dict:
    kv = {"general.architecture": arch, f"{arch}.block_count": LAYERS, f"{arch}.embedding_length": EMB,
          f"{arch}.attention.head_count": HEADS, f"{arch}.attention.head_count_kv": KV_HEADS,
          f"{arch}.attention.key_length": KEY, f"{arch}.expert_count": N_EXP, f"{arch}.expert_used_count": USED}
    kv.update({f"{arch}.{k}": v for k, v in over.items()})
    return kv


DENSE_FF = 60000                                             # a dense layer as heavy as a 27B-class model's


def layer(i: int, moe=True, gu="Q4_K", down="Q6_K") -> list:
    t = [(f"blk.{i}.attn_q.weight", [EMB, HEADS * KEY], "Q4_K"), (f"blk.{i}.attn_k.weight", [EMB, KV_HEADS * KEY], "Q4_K"),
         (f"blk.{i}.attn_v.weight", [EMB, KV_HEADS * KEY], "Q4_K"), (f"blk.{i}.attn_output.weight", [HEADS * KEY, EMB], "Q4_K"),
         (f"blk.{i}.attn_norm.weight", [EMB], "F32")]
    if moe:
        t += [(f"blk.{i}.ffn_gate_inp.weight", [EMB, N_EXP], "F32"), (f"blk.{i}.ffn_gate_exps.weight", [EMB, FF, N_EXP], gu),
              (f"blk.{i}.ffn_up_exps.weight", [EMB, FF, N_EXP], gu), (f"blk.{i}.ffn_down_exps.weight", [FF, EMB, N_EXP], down)]
    else:
        t += [(f"blk.{i}.ffn_gate.weight", [EMB, DENSE_FF], "Q4_K"), (f"blk.{i}.ffn_up.weight", [EMB, DENSE_FF], "Q4_K"),
              (f"blk.{i}.ffn_down.weight", [DENSE_FF, EMB], "Q6_K")]
    return t


def ends() -> list:
    return [("token_embd.weight", [EMB, VOCAB], "Q4_K"), ("output.weight", [EMB, VOCAB], "Q6_K"),
            ("output_norm.weight", [EMB], "F32")]


def model(moe=True, layers=LAYERS, meta=None, **kw) -> bytes:
    return T.gguf(meta or moe_meta(), ends() + [x for i in range(layers) for x in layer(i, moe, **kw)])


class Opts:
    """The command-line options plan_model reads."""
    ctx, kv, kv_gib, reserve_vram, reserve_ram, ram_bw = 32768, "f16", None, 1.5, 8.0, None
    threads, host, port, alias = None, "127.0.0.1", 8080, None

    def __init__(self, **kw):
        self.__dict__.update(kw)


def flag(flags: list, name: str):
    return flags[flags.index(name) + 1] if name in flags else None


class Planning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.src = M.Source(None, "https://huggingface.co", 5)

    def plan(self, data: bytes, vram=16.0, ram=32.0, name="m.gguf", **kw):
        (self.dir / name).write_bytes(data)
        files, names, absent = M.read_headers(str(self.dir / name), self.src)
        return L.plan_model(name, files, len(names), absent, {"vram": vram, "ram": ram}, Opts(**kw))

    def test_a_30b_moe_on_16_gib_of_vram_and_32_of_ram(self):
        p = self.plan(model())
        f = p.facts
        self.assertEqual((p.status, p.problems), ("ok", []))
        self.assertEqual((f["layers"], f["experts"], f["experts_used"], f["moe"]), (LAYERS, N_EXP, USED, True))
        self.assertEqual(p.ngl, L.ALL_LAYERS)
        self.assertTrue(f["output_on_gpu"])
        self.assertTrue(0 < p.n_cpu_moe < LAYERS)                       # some layers' experts fit the GPU, not all
        self.assertEqual(f["expert_layers_on_gpu"], LAYERS - p.n_cpu_moe)
        self.assertLessEqual(f["gpu_gib"], 16 - 1.5 + 0.1)              # inside the VRAM beside the reserve
        # every byte is somewhere: the weights and the KV cache sit on the GPU or in RAM
        self.assertAlmostEqual(f["gpu_gib"] + f["cpu_gib"], f["weights_gib"] + f["kv_gib"], delta=0.15)
        self.assertEqual(flag(p.flags, "--n-cpu-moe"), str(p.n_cpu_moe))
        self.assertEqual(flag(p.flags, "-ngl"), "999")
        self.assertIn("--jinja", p.flags)
        self.assertEqual((flag(p.flags, "--host"), flag(p.flags, "--port")), ("127.0.0.1", "8080"))   # Strata's address
        self.assertNotIn("--n-cpu-moe", p.flags_ot)
        self.assertEqual(flag(p.flags_ot, "-ot"), L.override_tensor(p.n_cpu_moe, LAYERS))

    def test_the_kv_cache_follows_the_header_arithmetic(self):
        p = self.plan(model(), ctx=32768)
        expect = LAYERS * KV_HEADS * (KEY + KEY) * 32768 * 2 / GIB      # f16: K and V of every layer
        self.assertAlmostEqual(p.facts["kv_gib"], expect, delta=0.05)
        q = self.plan(model(), ctx=32768, kv="q8_0")
        self.assertAlmostEqual(q.facts["kv_gib"], expect * 34 / 64, delta=0.05)
        self.assertEqual((flag(q.flags, "-ctk"), flag(q.flags, "-ctv")), ("q8_0", "q8_0"))
        self.assertNotIn("-ctk", p.flags)
        self.assertGreaterEqual(q.facts["expert_layers_on_gpu"], p.facts["expert_layers_on_gpu"])   # room won back

    def test_all_experts_fit_a_big_card(self):
        p = self.plan(model(), vram=48.0, ram=64.0)
        self.assertEqual((p.status, p.n_cpu_moe), ("ok", 0))
        self.assertNotIn("--n-cpu-moe", p.flags)
        self.assertNotIn("-ot", p.flags_ot)
        self.assertEqual(p.facts["expert_layers_on_gpu"], LAYERS)

    def test_a_small_card_keeps_all_experts_in_ram_and_part_of_the_dense_layers_off_the_gpu(self):
        p = self.plan(model(), vram=3.5, ram=64.0)
        self.assertLess(p.ngl, LAYERS)
        self.assertFalse(p.facts["output_on_gpu"])
        self.assertEqual(p.n_cpu_moe, LAYERS)
        self.assertEqual(p.facts["expert_layers_on_gpu"], 0)
        self.assertEqual(flag(p.flags_ot, "-ot"), r"\.ffn_.*_exps\.=CPU")
        self.assertTrue(any("do not all fit" in n for n in p.notes))

    def test_ram_verdicts(self):
        base = self.plan(model(), ram=64.0).facts["cpu_gib"]
        self.assertEqual(self.plan(model(), ram=64.0).status, "ok")
        tight = self.plan(model(), ram=base + 4)                        # fits, but not with 8 GiB for the system
        self.assertEqual(tight.status, "tight")
        self.assertTrue(any("RAM is tight" in n for n in tight.notes))
        slow = self.plan(model(), ram=base / 2)
        self.assertEqual(slow.status, "slow")
        self.assertTrue(any("SSD" in n for n in slow.notes))

    def test_a_dense_model_has_no_expert_flags(self):
        p = self.plan(model(moe=False, meta=moe_meta("qwen3", expert_count=0, expert_used_count=0)), vram=8.0)
        self.assertFalse(p.facts["moe"])
        self.assertIsNone(p.n_cpu_moe)
        self.assertNotIn("--n-cpu-moe", p.flags)
        self.assertNotIn("-ot", p.flags_ot)
        self.assertTrue(0 < p.ngl < LAYERS)                             # 8 GiB cannot hold it: part of the layers go
        self.assertEqual(p.status, "ok")
        self.assertIsNone(p.facts["expert_layers_on_gpu"])

    def test_hybrid_models_cache_kv_for_their_attention_layers_only(self):
        full = self.plan(model()).facts["kv_gib"]
        hybrid = self.plan(model(meta=moe_meta(full_attention_interval=4))).facts["kv_gib"]
        self.assertAlmostEqual(hybrid, full / 4, delta=0.1)                # (the facts are rounded to 0.1 GiB)
        table = [KV_HEADS if i % 4 == 3 else 0 for i in range(LAYERS)]
        self.assertAlmostEqual(self.plan(model(meta=moe_meta(**{"attention.head_count_kv": table}))).facts["kv_gib"],
                               full / 4, delta=0.1)
        ssm = self.plan(model(meta={**moe_meta(), "qwen3moe.ssm.state_size": 128}))
        self.assertEqual(ssm.facts["kv_gib"], full)
        self.assertTrue(any("upper bound" in n for n in ssm.notes))

    def test_a_kv_cache_the_header_cannot_size(self):
        kv = {k: v for k, v in moe_meta().items() if "head_count" not in k and "key_length" not in k}
        p = self.plan(model(meta=kv))
        self.assertEqual(p.status, "no")
        self.assertIn("--kv-gib", p.problems[0])
        q = self.plan(model(meta=kv), kv_gib=3.0)
        self.assertEqual((q.status, q.facts["kv_gib"]), ("ok", 3.0))

    def test_what_cannot_be_planned(self):
        self.assertEqual(self.plan(T.gguf({"split.no": 1}, [])).status, "no")             # not a first shard
        (self.dir / "s-00001-of-00002.gguf").write_bytes(model())
        files, names, absent = M.read_headers(str(self.dir / "s-00001-of-00002.gguf"), self.src)
        p = L.plan_model("s", files, len(names), absent, {"vram": 16.0, "ram": 32.0}, Opts())
        self.assertEqual(p.status, "no")
        self.assertIn("1 of 2 shards found", p.problems[0])
        odd = self.plan(T.gguf(moe_meta(), ends() + layer(0) + [("blk.0.x.weight", [256], "TQ1_0")]))
        self.assertEqual(odd.status, "no")
        self.assertIn("TQ1_0", odd.problems[0])                         # a type whose size this tool does not know

    def test_flash_next_is_pointed_at_strata(self):
        p = self.plan(T.flash_next())
        self.assertTrue(any("Strata's own engine" in n for n in p.notes))

    def test_the_roofline(self):
        p = self.plan(model(), ram_bw=50.0)
        f = p.facts
        self.assertIsNotNone(p.ceiling_tok_s)
        cpu_experts = (LAYERS - f["expert_layers_on_gpu"]) * (f["experts_gib"] / LAYERS) * USED / N_EXP
        self.assertAlmostEqual(p.ceiling_tok_s, 50e9 / (cpu_experts * GIB), delta=p.ceiling_tok_s * 0.1)
        self.assertIsNone(self.plan(model()).ceiling_tok_s)              # not asked for
        self.assertIsNone(self.plan(model(), vram=48.0, ram=64.0, ram_bw=50.0).ceiling_tok_s)   # nothing read by the CPU

    def test_override_tensor_selects_exactly_the_first_layers(self):
        pat = L.override_tensor(3, LAYERS).removesuffix("=CPU")
        hit = lambda n: bool(re.search(pat, n))
        self.assertTrue(all(hit(f"blk.{i}.{r}_exps.weight") for i in range(3) for r in ("ffn_gate", "ffn_up", "ffn_down")))
        self.assertFalse(any(hit(f"blk.{i}.ffn_up_exps.weight") for i in (3, 12, 20, 30, 47)))
        self.assertFalse(hit("blk.1.attn_q.weight"))                      # only the experts
        self.assertEqual(L.override_tensor(LAYERS, LAYERS), r"\.ffn_.*_exps\.=CPU")

    def test_fit_from_end(self):
        self.assertEqual(L.fit_from_end([5, 1, 1, 1], 3), 3)
        self.assertEqual(L.fit_from_end([1, 1, 4], 3), 0)                 # the last layer must fit first
        self.assertEqual(L.fit_from_end([1, 1], 100), 2)
        self.assertEqual(L.fit_from_end([1, 1], -1), 0)


class Server(unittest.TestCase):
    """build_command: the flags spelled for the build whose --help it is given."""
    MODERN = "--n-cpu-moe N\n -fa, --flash-attn [on|off|auto]\n"

    def plan(self, kv="f16"):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            (Path(d) / "m.gguf").write_bytes(model())
            files, names, absent = M.read_headers(str(Path(d) / "m.gguf"), M.Source(None, "https://huggingface.co", 5))
            return L.plan_model("m", files, 1, absent, {"vram": 16.0, "ram": 32.0}, Opts(kv=kv))

    def test_current_and_older_builds(self):
        p = self.plan()
        modern = L.build_command("llama-server", "/m.gguf", p, self.MODERN)
        self.assertEqual(modern[:3], ["llama-server", "-m", "/m.gguf"])
        self.assertIn("--n-cpu-moe", modern)
        self.assertNotIn("-ot", modern)
        old = L.build_command("llama-server", "/m.gguf", p, "-ot, --override-tensor <tensor name pattern>=<buffer type>\n")
        self.assertIn("-ot", old)
        self.assertNotIn("--n-cpu-moe", old)
        self.assertEqual(L.build_command("llama-server", "/m.gguf", p, ""), modern)   # help unreadable: a current build
        self.assertNotIn("-fa", modern)                                   # f16 cache: no flash attention needed

    def test_a_quantized_cache_gets_flash_attention_in_the_builds_spelling(self):
        p = self.plan("q8_0")
        self.assertEqual(L.build_command("s", "/m", p, self.MODERN)[-2:], ["-fa", "on"])
        self.assertEqual(L.build_command("s", "/m", p, "-fa, --flash-attn      enable flash attention\n--n-cpu-moe")[-1], "-fa")
        self.assertNotIn("-fa", L.build_command("s", "/m", p, ""))

    def test_join_quotes_what_a_shell_would_split(self):
        self.assertIn("'blk\\.(0|1)\\.ffn_.*_exps\\.=CPU'", L.join(["-ot", r"blk\.(0|1)\.ffn_.*_exps\.=CPU"]))


@unittest.skipIf(sys.platform == "win32", "the fake llama-server is a POSIX script")
class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "m.gguf").write_bytes(model())
        self.record = self.dir / "argv.json"
        self.server = self.dir / "llama-server"
        self.server.write_text(f"#!{sys.executable}\nimport json, sys\n"
                               f"if '--help' in sys.argv: print('--n-cpu-moe N'); print('-fa, --flash-attn [on|off|auto]')\n"
                               f"else: json.dump(sys.argv[1:], open({str(self.record)!r}, 'w'))\n")
        self.server.chmod(self.server.stat().st_mode | stat.S_IXUSR)

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = L.main(["--vram", "16", "--ram", "32", *argv])
        except SystemExit as e:                                           # argparse usage errors
            code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_plan_json(self):
        code, out, _ = self.cli(str(self.dir / "m.gguf"), "--json", "--ctx", "8192", "--alias", "muse")
        plan = json.loads(out)[0]
        self.assertEqual((code, plan["status"], plan["facts"]["ctx"]), (0, "ok", 8192))
        self.assertEqual(plan["model_path"], str((self.dir / "m.gguf").resolve()))
        self.assertEqual(flag(plan["flags"], "--alias"), "muse")

    def test_plan_text(self):
        code, out, _ = self.cli(str(self.dir / "m.gguf"), "--ram-bw", "50")
        self.assertEqual(code, 0)
        for word in ("OK", "qwen3moe", "128 experts (8 used)", "on the GPU", "in RAM", "ceiling", "--n-cpu-moe", "older llama"):
            self.assertIn(word, out)

    def test_run_starts_the_server_with_the_plan(self):
        code, out, _ = self.cli(str(self.dir / "m.gguf"), "--run", "--llama-server", str(self.server), "--port", str(free_port()),
                                "--kv", "q8_0")
        self.assertEqual(code, 0, out)
        argv = json.loads(self.record.read_text())
        self.assertEqual(argv[:2], ["-m", str((self.dir / "m.gguf").resolve())])
        self.assertIn("--n-cpu-moe", argv)
        self.assertEqual(argv[-2:], ["-fa", "on"])
        self.assertIn("API: http://127.0.0.1:", out)

    def test_run_refuses(self):
        port = free_port()
        with socket.socket() as busy:                                     # something already listens: Strata, say
            busy.bind(("127.0.0.1", port))
            busy.listen()
            code, _, err = self.cli(str(self.dir / "m.gguf"), "--run", "--llama-server", str(self.server), "--port", str(port))
        self.assertEqual(code, 1)
        self.assertIn("is in use", err)
        self.assertFalse(self.record.exists())
        code, _, err = self.cli(str(self.dir / "m.gguf"), "--run", "--llama-server", str(self.dir / "nope"), "--port", str(free_port()))
        self.assertEqual(code, 2)
        self.assertIn("llama-server not found", err)
        code, _, _ = self.cli("owner/repo", "--run")
        self.assertEqual(code, 2)                                         # --run needs one local file

    def test_a_slow_plan_needs_force(self):
        code, _, err = self.cli(str(self.dir / "m.gguf"), "--run", "--llama-server", str(self.server), "--ram", "6",
                                "--port", str(free_port()))
        self.assertEqual(code, 1)
        self.assertIn("SLOW", err)
        self.assertFalse(self.record.exists())
        code, _, _ = self.cli(str(self.dir / "m.gguf"), "--run", "--force", "--llama-server", str(self.server), "--ram", "6",
                              "--port", str(free_port()))
        self.assertEqual(code, 0)
        self.assertTrue(self.record.exists())

    def test_usage_errors(self):
        for argv in (["--ctx", "0"], ["--vram", "-1"], ["--threads", "0"], ["--reserve-ram", "-1"]):
            self.assertEqual(self.cli(str(self.dir / "m.gguf"), *argv)[0], 2, argv)
        self.assertEqual(self.cli(str(self.dir / "missing.gguf"))[0], 2)

    def test_a_repo_over_http(self):
        hub = T.FakeHub({"Muse-Q4_K_M.gguf": model(), "mmproj-x.gguf": T.gguf({"general.architecture": "clip"}, [])})
        self.addCleanup(hub.close)
        code, out, _ = self.cli("owner/repo", "--endpoint", hub.url, "--json")
        plans = json.loads(out)
        self.assertEqual((code, len(plans), plans[0]["status"], plans[0]["model_path"]), (0, 1, "ok", None))
        self.assertEqual(plans[0]["target"], "owner/repo :: Muse-Q4_K_M.gguf")


class Detection(unittest.TestCase):
    def test_ram(self):
        self.assertGreater(L.detect_ram_gib() or 0, 0.5)

    def test_vram_from_nvidia_smi(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            exe = Path(d) / "nvidia-smi"
            exe.write_text(f"#!{sys.executable}\nprint('16384')\nprint('8192')\n")
            exe.chmod(0o755)
            old = L.shutil.which
            L.shutil.which = lambda n: str(exe) if n == "nvidia-smi" else None
            try:
                self.assertEqual(L.detect_vram_gib(), 16.0)
                L.shutil.which = lambda n: None
                self.assertIsNone(L.detect_vram_gib())
            finally:
                L.shutil.which = old


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
