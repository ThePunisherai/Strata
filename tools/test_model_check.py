"""model_check's verdicts over synthetic GGUF headers (no model, no GPU, no network): the architecture guard, the
geometry, the expert types, split models, and the HTTP path - Range windows, the Hub's redirect, and the token that
must not follow it - against a local server.

    python -m unittest tools.test_model_check
"""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gguf_reader as G  # noqa: E402
import model_check as M  # noqa: E402

TYPE_ID = {v: k for k, v in G.GGML_TYPES.items()}
LAYERS, H, FF = 48, 2560, 640


def _s(x: str) -> bytes:
    return struct.pack("<Q", len(x.encode())) + x.encode()


def gguf(kv: dict, tensors: list) -> bytes:
    """A GGUF v3 header: kv = {key: str | int (u32) | [int] (i32 array)}, tensors = [(name, shape, type name)]; no tensor data."""
    b = bytearray(struct.pack("<IIQQ", G.GGUF_MAGIC, 3, len(tensors), len(kv)))
    for k, v in kv.items():
        if isinstance(v, str):
            b += _s(k) + struct.pack("<I", 8) + _s(v)
        elif isinstance(v, list):                                   # an array of i32
            b += _s(k) + struct.pack("<IIQ", 9, 5, len(v)) + struct.pack(f"<{len(v)}i", *v)
        else:
            b += _s(k) + struct.pack("<II", 4, v)
    for name, shape, ty in tensors:
        b += _s(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape) + \
            struct.pack("<IQ", TYPE_ID[ty], 0)
    return bytes(b)


def meta(arch="qwen4exp", **over) -> dict:
    kv = {"general.architecture": arch, f"{arch}.block_count": LAYERS, f"{arch}.embedding_length": H,
          f"{arch}.expert_count": 512, f"{arch}.expert_used_count": 10, f"{arch}.attention.head_count": 24,
          f"{arch}.attention.head_count_kv": 2}
    kv.update({f"{arch}.{k}": v for k, v in over.items()})
    return kv


def layer_tensors(layers, n_exp=512, gu="IQ3_XXS", down="IQ4_NL") -> list:
    t = []
    for i in layers:
        t += [(f"blk.{i}.ffn_gate_exps.weight", [H, FF, n_exp], gu), (f"blk.{i}.ffn_up_exps.weight", [H, FF, n_exp], gu),
              (f"blk.{i}.ffn_down_exps.weight", [FF, H, n_exp], down),
              (f"blk.{i}.ffn_gate_inp.weight", [H, n_exp], "F32")]
    return t


def model_tensors() -> list:
    return [("token_embd.weight", [H, 1000], "Q8_0"), ("per_layer_token_embd.weight", [160, 1000], "Q5_0"),
            ("output_hc_down.weight", [H, 320], "BF16")]


def flash_next(**kw) -> bytes:
    return gguf(meta(**kw.pop("meta", {})), model_tensors() + layer_tensors(range(LAYERS), **kw))


class Verdicts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        M._bf16_wanted = lambda: ({"ffn_gate_inp.weight": "BF16", "hc_attn_down.weight": "BF16"}, {"Q2_0", "Q8_0"})

    def check(self, data: bytes, name="m.gguf"):
        p = self.dir / name
        p.write_bytes(data)
        return M.read_model(name, str(p), M.Source(None, "https://huggingface.co", 5))

    def test_flash_next_layout_runs(self):
        r = self.check(flash_next())
        self.assertEqual((r.status, r.problems), ("ok", []))
        self.assertEqual(r.facts["variant"], "the original model")
        self.assertEqual(r.facts["expert_types"], {"gate_up": ["IQ3_XXS"], "down": ["IQ4_NL"]})
        per_layer = 2 * (H * FF * 512 // 256 * 98) + FF * H * 512 // 32 * 18     # IQ3_XXS gate+up, IQ4_NL down
        self.assertEqual(r.facts["experts_gib"], round(LAYERS * per_layer / 2**30, 1))
        self.assertEqual(r.facts["bf16_conversions"], LAYERS)   # the F32 routers: the engine reads BF16
        self.assertTrue(any("--compat-bf16" in line for line in r.next))

    def test_coder_is_recognised(self):
        r = self.check(flash_next(n_exp=256, meta={"expert_count": 256}))
        self.assertEqual((r.status, r.facts["variant"]), ("ok", "the Coder (half the experts)"))

    def test_other_architecture_is_refused_with_its_numbers(self):
        for arch in ("qwen3moe", "deepseek2", "llama"):
            r = self.check(gguf(meta(arch), [("token_embd.weight", [2048, 10], "Q4_K")]))
            self.assertEqual(r.status, "no")
            self.assertEqual(r.architecture, arch)
            self.assertIn(f"'{arch}'", r.problems[0])
            self.assertEqual(r.facts["block_count"], LAYERS)

    def test_no_architecture_is_not_a_first_shard(self):
        r = self.check(gguf({"split.no": 1}, []))
        self.assertEqual(r.status, "no")
        self.assertIn("general.architecture", r.problems[0])

    def test_geometry_the_kernels_are_not_built_for(self):
        for key, value in (("embedding_length", 2048), ("block_count", 40), ("attention.head_count", 32),
                           ("expert_used_count", 8)):
            r = self.check(gguf(meta(**{key: value}), model_tensors() + layer_tensors(range(LAYERS))))
            self.assertEqual(r.status, "no", key)
            self.assertTrue(any(key in p or "used per token" in p for p in r.problems), (key, r.problems))

    def test_missing_geometry_key_is_named(self):
        kv = meta()
        del kv["qwen4exp.attention.head_count_kv"]
        r = self.check(gguf(kv, model_tensors() + layer_tensors(range(LAYERS))))
        self.assertEqual(r.status, "no")
        self.assertIn("missing qwen4exp.attention.head_count_kv", r.problems)

    def test_expert_types(self):
        r = self.check(flash_next(down="Q5_0"))
        self.assertEqual(r.status, "no")
        self.assertTrue(any("Q5_0" in p and "refuses" in p for p in r.problems))
        r = self.check(flash_next(gu="Q4_K", down="Q5_1"))
        self.assertEqual((r.status, r.problems), ("check", []))
        self.assertEqual(len(r.warnings), 1)                    # one condition, both roles named
        self.assertIn("gate/up Q4_K, down Q5_1", r.warnings[0])
        r = self.check(flash_next(gu="Q6_K"))
        self.assertEqual(r.status, "no")
        self.assertIn("not a type", r.problems[0])

    def test_gate_and_up_must_agree(self):
        t = layer_tensors(range(LAYERS))
        t[1] = (t[1][0], t[1][1], "IQ2_XS")
        r = self.check(gguf(meta(), model_tensors() + t))
        self.assertEqual(r.status, "no")
        self.assertTrue(any("differ in type" in p for p in r.problems))

    def test_expert_shape_is_checked(self):
        t = layer_tensors(range(LAYERS))
        t[0] = (t[0][0], [H, 512, 512], "IQ3_XXS")
        r = self.check(gguf(meta(), model_tensors() + t))
        self.assertEqual(r.status, "no")
        self.assertTrue(any("shape" in p for p in r.problems))

    def test_missing_layer_tensors(self):
        r = self.check(gguf(meta(), model_tensors() + layer_tensors(range(LAYERS - 1))))
        self.assertEqual(r.status, "no")
        self.assertTrue(any("missing" in p and "blk.47." in p for p in r.problems))

    def test_unusual_expert_count_is_a_note(self):
        r = self.check(flash_next(n_exp=384, meta={"expert_count": 384}))
        self.assertEqual((r.status, r.problems), ("check", []))
        self.assertIn("384 experts", r.warnings[0])

    def test_split_model_is_read_across_its_shards(self):
        half = LAYERS // 2
        (self.dir / "m-00001-of-00002.gguf").write_bytes(gguf(meta(), model_tensors() + layer_tensors(range(half))))
        (self.dir / "m-00002-of-00002.gguf").write_bytes(gguf({"split.no": 1}, layer_tensors(range(half, LAYERS))))
        src = M.Source(None, "https://huggingface.co", 5)
        r = M.read_model("m", str(self.dir / "m-00001-of-00002.gguf"), src)
        self.assertEqual((r.status, r.shards, r.shards_read), ("ok", 2, 2))
        # reading shard 2's name finds the same model from shard 1
        self.assertEqual(M.read_model("m", str(self.dir / "m-00002-of-00002.gguf"), src).status, "ok")
        (self.dir / "m-00002-of-00002.gguf").unlink()
        r = M.read_model("m", str(self.dir / "m-00001-of-00002.gguf"), src)
        self.assertEqual((r.status, r.shards_read), ("check", 1))      # the others' tensors are not judged
        self.assertTrue(any("shards not found" in w for w in r.warnings))

    def test_a_missing_file_and_a_non_gguf(self):
        with self.assertRaises(FileNotFoundError):
            M.read_model("x", str(self.dir / "nope.gguf"), M.Source(None, "https://huggingface.co", 5))
        with self.assertRaises(ValueError):
            self.check(b"not a gguf at all, but long enough to be read as a header....")


class Targets(unittest.TestCase):
    def test_classify(self):
        c = lambda a: M.classify(a, "main", "https://huggingface.co")
        self.assertEqual(c("owner/repo"), ("repo", ("owner/repo", "main")))
        self.assertEqual(c("https://huggingface.co/owner/repo"), ("repo", ("owner/repo", "main")))
        self.assertEqual(c("https://huggingface.co/owner/repo/tree/abc123"), ("repo", ("owner/repo", "abc123")))
        self.assertEqual(c("https://huggingface.co/owner/repo/blob/main/Q4/m-00001-of-00002.gguf"),
                         ("url", "https://huggingface.co/owner/repo/resolve/main/Q4/m-00001-of-00002.gguf"))
        self.assertEqual(c("https://example.org/a/b.gguf"), ("url", "https://example.org/a/b.gguf"))
        with self.assertRaises(ValueError):
            c("not a target")

    def test_the_token_goes_to_the_hub_only(self):
        for host, ours in (("huggingface.co", True), ("cdn-lfs.hf.co", True), ("hf.co", True),
                           ("nothf.co", False), ("huggingface.co.evil.org", False), ("example.org", False)):
            self.assertEqual(M.is_hub(host), ours, host)
        src = M.Source("secret", "https://huggingface.co", 5)
        self.assertEqual(src._headers("https://huggingface.co/api/models/a/b"), {"Authorization": "Bearer secret"})
        self.assertEqual(src._headers("https://nothf.co/x"), {})

    def test_shard_names(self):
        self.assertEqual(M.shard_names("m-00002-of-00003.gguf"),
                         ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"])
        self.assertEqual(M.shard_names("m.gguf"), ["m.gguf"])


class FakeHub:
    """A Hub (the API and `resolve`, which redirects) and a CDN (Range) on two local ports; records the headers."""

    def __init__(self, files: dict, ignore_range=False):
        self.files, self.seen, self.ignore_range = files, [], ignore_range
        outer = self

        class Cdn(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append(("cdn", self.path, dict(self.headers)))
                data = outer.files.get(self.path.rsplit("/", 1)[-1])
                if data is None:
                    return self.send_error(404)
                rng = self.headers.get("Range")
                if rng and not outer.ignore_range:
                    a, b = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                    if a >= len(data):
                        return self.send_error(416)
                    part = data[a:b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{a + len(part) - 1}/{len(data)}")
                    body = part
                else:
                    self.send_response(200)
                    body = data
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class Hub(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append(("hub", self.path, dict(self.headers)))
                if self.path == "/api/models/owner/repo":
                    body = json.dumps({"siblings": [{"rfilename": n} for n in sorted(outer.files)] +
                                       [{"rfilename": "README.md"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                if "/resolve/" in self.path:
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{outer.cdn.server_port}/blob/{self.path.rsplit('/', 1)[-1]}")
                    self.end_headers()
                    return
                self.send_error(404)

        self.cdn = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Cdn)
        self.hub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
        for s in (self.cdn, self.hub):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.hub.server_port}"

    def close(self):
        for s in (self.cdn, self.hub):
            s.shutdown()
            s.server_close()


class Remote(unittest.TestCase):
    def setUp(self):
        half = LAYERS // 2
        self.files = {
            "Good-IQ3_XXS-00001-of-00002.gguf": gguf(meta(), model_tensors() + layer_tensors(range(half))),
            "Good-IQ3_XXS-00002-of-00002.gguf": gguf({"split.no": 1}, layer_tensors(range(half, LAYERS))),
            "Good-Q2_0.gguf": flash_next(gu="Q2_0", down="Q2_0"),
            "Other-Q4_K_M.gguf": gguf(meta("qwen3moe"), [("token_embd.weight", [2048, 10], "Q4_K")]),
            "mmproj-Good-BF16.gguf": gguf({"general.architecture": "clip"}, []),
        }
        self.hub = FakeHub(self.files)
        self.addCleanup(self.hub.close)
        self.run_cli = lambda *argv: self._cli(*argv)
        M._bf16_wanted = lambda: (None, set())

    def _cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = M.main(["--endpoint", self.hub.url, "--token", "secret", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_a_whole_repo_over_http(self):
        code, out, _ = self._cli("owner/repo", "--json")
        reports = {r["target"].split(" :: ")[1]: r for r in json.loads(out)}
        self.assertEqual(set(reports), {"Good-IQ3_XXS-00001-of-00002.gguf", "Good-Q2_0.gguf", "Other-Q4_K_M.gguf"})
        self.assertEqual(reports["Good-IQ3_XXS-00001-of-00002.gguf"]["status"], "ok")
        self.assertEqual(reports["Good-IQ3_XXS-00001-of-00002.gguf"]["shards_read"], 2)
        self.assertEqual(reports["Good-Q2_0.gguf"]["status"], "ok")
        self.assertEqual(reports["Other-Q4_K_M.gguf"]["status"], "no")
        self.assertEqual(code, 1)

    def test_the_token_reaches_the_hub_and_not_the_cdn(self):
        self._cli("owner/repo", "--match", "Q2_0")
        hub = [h for who, _, h in self.hub.seen if who == "hub"]
        cdn = [h for who, _, h in self.hub.seen if who == "cdn"]
        self.assertTrue(hub and cdn)
        self.assertTrue(all(h.get("Authorization") == "Bearer secret" for h in hub))
        self.assertFalse(any("Authorization" in h for h in cdn))
        self.assertTrue(all(h.get("Range", "").startswith("bytes=0-") for h in cdn))   # a header, not the model

    def test_match_and_exit_status(self):
        code, out, _ = self._cli("owner/repo", "--match", "Q2_0")
        self.assertEqual(code, 0)
        self.assertIn("OK", out)
        self.assertNotIn("Other-Q4_K_M", out)
        code, out, _ = self._cli("owner/repo", "--match", "nothing-like-this")
        self.assertEqual(code, 2)
        self.assertIn("no model .gguf file matching", out)

    def test_a_file_url(self):
        code, out, _ = self._cli(f"{self.hub.url}/owner/repo/resolve/main/Other-Q4_K_M.gguf")
        self.assertEqual(code, 1)
        self.assertIn("'qwen3moe'", out)

    def test_http_errors_are_named(self):
        code, out, _ = self._cli("owner/missing")
        self.assertEqual(code, 2)
        self.assertIn("HTTP 404", out)

    def test_headers_larger_than_one_window_are_read_in_several(self):
        old = M.CHUNK
        M.CHUNK = 4096
        self.addCleanup(setattr, M, "CHUNK", old)
        url = f"{self.hub.url}/owner/repo/resolve/main/Good-Q2_0.gguf"
        remote = M.Source(None, self.hub.url, 5).open_url(url)
        p = Path(tempfile.mkdtemp()) / "x.gguf"
        p.write_bytes(self.files["Good-Q2_0.gguf"])
        local = G.GGUFFile(p)
        self.assertEqual(remote.metadata, local.metadata)
        self.assertEqual([(t.name, t.shape, t.type_name) for t in remote.tensors],
                         [(t.name, t.shape, t.type_name) for t in local.tensors])
        self.assertGreater(len([1 for who, _, h in self.hub.seen if who == "cdn"]), 2)

    def test_a_server_that_ignores_range_is_refused(self):
        self.hub.ignore_range = True
        old = M.CHUNK
        M.CHUNK = 4096
        self.addCleanup(setattr, M, "CHUNK", old)
        with self.assertRaisesRegex(OSError, "ignores Range"):
            M.Source(None, self.hub.url, 5).open_url(f"{self.hub.url}/owner/repo/resolve/main/Good-Q2_0.gguf")


if __name__ == "__main__":
    unittest.main()
