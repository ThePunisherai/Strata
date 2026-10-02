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
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gguf_reader as G  # noqa: E402
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


def ends(tied=False) -> list:
    out = [] if tied else [("output.weight", [EMB, VOCAB], "Q6_K")]
    return [("token_embd.weight", [EMB, VOCAB], "Q4_K"), *out, ("output_norm.weight", [EMB], "F32")]


def model(moe=True, layers=LAYERS, meta=None, tied=False, **kw) -> bytes:
    return T.gguf(meta or moe_meta(), ends(tied) + [x for i in range(layers) for x in layer(i, moe, **kw)])


class Opts:
    """The command-line options plan_model reads."""
    ctx, kv, kv_gib, reserve_vram, reserve_ram, ram_bw = 32768, "f16", None, 1.5, 8.0, None
    threads, host, port, alias, parallel = None, "127.0.0.1", 8080, None, 1

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
        self.assertEqual(flag(p.flags, "-np"), "1")                      # one request at a time, like Strata
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

    def test_sliding_window_layers_cache_the_window_not_the_context(self):
        full = self.plan(model()).facts["kv_gib"]
        no_pattern = self.plan(model(meta=moe_meta(**{"attention.sliding_window": 1024})))
        self.assertEqual(no_pattern.facts["kv_gib"], full)               # no layer table: an upper bound, and it says so
        self.assertTrue(any("sliding-window attention without a layer pattern" in n for n in no_pattern.notes))
        every_6th_global = self.plan(model(meta=moe_meta(**{"attention.sliding_window": 1024, "attention.sliding_window_pattern": 6})))
        swa_layers = sum(1 for i in range(LAYERS) if i % 6 < 5)
        window = 1536                                                    # 1024 + the 512-token micro-batch, already a multiple of 256
        expect = (LAYERS - swa_layers) * 32768 + swa_layers * window
        self.assertAlmostEqual(every_6th_global.facts["kv_gib"], full * expect / (LAYERS * 32768), delta=0.06)
        self.assertTrue(any(f"({swa_layers} of {LAYERS})" in n and f"{window} tokens" in n for n in every_6th_global.notes))
        flags = [i % 3 != 2 for i in range(LAYERS)]                      # a per-layer table, true = a sliding-window layer
        table = self.plan(model(meta=moe_meta(**{"attention.sliding_window": 1024, "attention.sliding_window_pattern": flags})))
        self.assertAlmostEqual(table.facts["kv_gib"], full * (flags.count(False) * 32768 + flags.count(True) * window)
                               / (LAYERS * 32768), delta=0.06)
        slots = self.plan(model(meta=moe_meta(**{"attention.sliding_window": 1024, "attention.sliding_window_pattern": 6})),
                          parallel=4)
        self.assertGreater(slots.facts["kv_gib"], every_6th_global.facts["kv_gib"])      # four windows per layer
        self.assertEqual(flag(slots.flags, "-np"), "4")
        small = self.plan(model(meta=moe_meta(**{"attention.sliding_window": 1024, "attention.sliding_window_pattern": 6})), ctx=512)
        self.assertLess(small.facts["kv_gib"], 0.1)                      # the window never exceeds the context

    def test_a_latent_attention_header_is_marked_an_upper_bound(self):
        p = self.plan(model(meta=moe_meta(**{"attention.kv_lora_rank": 512})))
        self.assertTrue(any("latent attention" in n for n in p.notes))

    def test_tied_embeddings_are_one_copy_in_ram_and_one_more_on_the_gpu(self):
        tied = model(tied=True)
        sizes = {t.name: t.expected_bytes() for t in M.G.GGUFFile(self.dir / "tied.gguf").tensors} \
            if (self.dir / "tied.gguf").write_bytes(tied) else {}
        on_disk = sum(sizes.values()) / GIB
        embed = sizes["token_embd.weight"] / GIB
        p = self.plan(tied)                                              # fits: the output layer takes a GPU copy
        self.assertAlmostEqual(p.facts["weights_gib"], on_disk, delta=0.06)       # the header's tensors, once
        self.assertTrue(p.facts["output_on_gpu"])
        self.assertAlmostEqual(p.facts["gpu_gib"] + p.facts["cpu_gib"], on_disk + p.facts["kv_gib"] + embed, delta=0.15)
        small = self.plan(tied, vram=3.5, ram=64.0)                      # the output layer stays on the CPU: the same pages
        self.assertFalse(small.facts["output_on_gpu"])
        self.assertAlmostEqual(small.facts["gpu_gib"] + small.facts["cpu_gib"], on_disk + small.facts["kv_gib"], delta=0.15)
        untied = self.plan(model())
        self.assertAlmostEqual(untied.facts["gpu_gib"] + untied.facts["cpu_gib"],
                               untied.facts["weights_gib"] + untied.facts["kv_gib"], delta=0.15)

    def test_a_quantized_cache_says_it_needs_flash_attention(self):
        q = self.plan(model(), kv="q8_0")
        self.assertTrue(any("-fa on" in n and "-fa`" in n for n in q.notes))                  # both spellings
        self.assertFalse(any("flash attention: add" in n for n in self.plan(model()).notes))

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
        self.assertEqual(plan["model_path"], os.path.abspath(self.dir / "m.gguf"))
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
        self.assertEqual(argv[:2], ["-m", os.path.abspath(self.dir / "m.gguf")])
        self.assertIn("--n-cpu-moe", argv)
        self.assertEqual(argv[-2:], ["-fa", "on"])
        self.assertIn("API: http://127.0.0.1:", out)

    def test_run_refuses(self):
        (self.dir / "empty").mkdir()
        with env(PATH=str(self.dir / "empty"), LLAMA_SERVER=None):         # a llama-server on the machine must not matter
            self._run_refuses()

    def _run_refuses(self):
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
        self.assertIn("not an executable file", err)                      # the path that was named, not another binary
        code, _, err = self.cli(str(self.dir / "m.gguf"), "--run", "--port", str(free_port()))
        self.assertEqual(code, 2)
        self.assertIn("llama-server not found", err)                      # nothing named, nothing on PATH
        for argv in (["owner/repo", "--run"], [str(self.dir / "m.gguf"), str(self.dir / "m.gguf"), "--run"]):
            code, _, err = self.cli(*argv, "--endpoint", "http://127.0.0.1:9")    # refused before anything is read or fetched
            self.assertEqual(code, 2)
            self.assertIn("--run needs exactly one target", err)

    def test_a_huggingface_cache_layout_keeps_its_symlink_names(self):
        blobs, snap = self.dir / "blobs", self.dir / "snapshots" / "abc"
        blobs.mkdir()
        snap.mkdir(parents=True)
        half = LAYERS // 2
        (blobs / "aa11").write_bytes(T.gguf(moe_meta(), ends() + [x for i in range(half) for x in layer(i)]))
        (blobs / "bb22").write_bytes(T.gguf({"split.no": 1}, [x for i in range(half, LAYERS) for x in layer(i)]))
        os.symlink("../../blobs/aa11", snap / "M-Q4-00001-of-00002.gguf")
        os.symlink("../../blobs/bb22", snap / "M-Q4-00002-of-00002.gguf")
        first = os.path.abspath(snap / "M-Q4-00001-of-00002.gguf")
        for given in ("M-Q4-00001-of-00002.gguf", "M-Q4-00002-of-00002.gguf"):          # llama.cpp needs shard 1's own name
            code, out, _ = self.cli(str(snap / given), "--json")
            self.assertEqual((code, json.loads(out)[0]["status"], json.loads(out)[0]["model_path"]), (0, "ok", first), given)
        code, _, _ = self.cli(str(snap / "M-Q4-00002-of-00002.gguf"), "--run", "--llama-server", str(self.server),
                              "--port", str(free_port()))
        self.assertEqual(json.loads(self.record.read_text())[:2], ["-m", first])

    def test_a_corrupt_file_is_an_error_and_the_good_plan_is_still_printed(self):
        (self.dir / "bad.gguf").write_bytes(struct.pack("<IIQQ", G.GGUF_MAGIC, 3, 0, 1) + struct.pack("<Q", 2**64 - 1))
        code, out, _ = self.cli(str(self.dir / "bad.gguf"), str(self.dir / "m.gguf"), "--json")
        self.assertEqual((code, [p["status"] for p in json.loads(out)]), (2, ["error", "ok"]))

    def test_a_server_path_that_is_named_is_never_replaced(self):
        path_dir = self.dir / "bin"
        path_dir.mkdir()
        (path_dir / "llama-server").write_bytes(self.server.read_bytes())
        (path_dir / "llama-server").chmod(0o755)
        notexec = self.dir / "notexec"
        notexec.write_text("#!/bin/sh\n")
        with env(PATH=str(path_dir), LLAMA_SERVER=None):
            for named in ("/typo/llama-server", str(notexec)):                     # a llama-server IS on PATH: still an error
                code, _, err = self.cli(str(self.dir / "m.gguf"), "--run", "--llama-server", named, "--port", str(free_port()))
                self.assertEqual(code, 2, named)
                self.assertIn("not an executable file", err)
                self.assertFalse(self.record.exists())
            with env(LLAMA_SERVER="/typo/from-env"):
                code, _, err = self.cli(str(self.dir / "m.gguf"), "--run", "--port", str(free_port()))
                self.assertEqual(code, 2)
                self.assertIn("$LLAMA_SERVER", err)
            code, _, _ = self.cli(str(self.dir / "m.gguf"), "--run", "--port", str(free_port()))      # nothing named: PATH
            self.assertEqual(code, 0)
            self.assertTrue(self.record.exists())

    def test_sigterm_stops_llama_server_too(self):
        import signal
        import subprocess
        import time
        pidfile = self.dir / "server.pid"
        sleeper = self.dir / "sleeper"
        sleeper.write_text(f"#!{sys.executable}\nimport os, sys, time\n"
                           f"if '--help' in sys.argv: print('--n-cpu-moe')\n"
                           f"else:\n    open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n    time.sleep(120)\n")
        sleeper.chmod(0o755)
        planner = subprocess.Popen([sys.executable, str(ROOT / "tools" / "llama_plan.py"), str(self.dir / "m.gguf"), "--vram", "16",
                                    "--ram", "32", "--run", "--llama-server", str(sleeper), "--port", str(free_port())],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.addCleanup(lambda: planner.poll() is None and planner.kill())
        for _ in range(100):                                              # the server has started
            if pidfile.exists() and pidfile.read_text():
                break
            time.sleep(0.1)
        server_pid = int(pidfile.read_text())
        planner.send_signal(signal.SIGTERM)
        planner.wait(timeout=30)
        for _ in range(100):
            try:
                os.kill(server_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(server_pid, signal.SIGKILL)
            self.fail("llama-server outlived the planner")

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

    @unittest.skipIf(sys.platform == "win32", "the fake nvidia-smi is a POSIX script")
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


@contextlib.contextmanager
def env(**kw):
    """Environment variables set (None = removed) for a block."""
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
