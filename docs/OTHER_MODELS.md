# Other models: what Strata can run

Strata runs **one architecture**, `qwen4exp` - the Qwen3.8-Flash-Next layout: 48 layers, width 2560, 24 attention
heads (2 KV), 512 routed experts of which 10 are used per token (256 in the Coder), hyper-connections, a PLE table
and gated-delta/QSA attention. Its speed on a small PC comes from that exact layout: the experts live in RAM, the
busiest ones on the GPU, and every kernel (GPU and CPU) is written for those dimensions. The engine says so itself:
`check_architecture` (`include/strata/artifact/gguf_reader.hpp`) refuses every other `general.architecture`.

So the question for any model file is: **is it a `qwen4exp` file?**

- **Yes** - a fine-tune, an abliteration or an "uncensored" release of Flash-Next keeps the layout. It runs, as
  Swift 1.5 and OrcaRouter's Uncensored already do. Setup lists only the families it was built for (qwen, swift,
  coder, unsloth); any other such file is packed by hand with `tools/iq_pack.py` ([docs/ORCA.md](ORCA.md)).
- **No** - a dense 3B or 27B model, another MoE (Qwen3-30B-A3B style, GLM, Kimi, DeepSeek style) or any other
  architecture. Making one of them run is a new model implementation inside the engine - a loader plus the
  attention, router and expert kernels of that architecture - not a setting, a conversion or a new download.
  Those files belong in a general-purpose runtime such as llama.cpp.

## Finding out in seconds: `tools/model_check.py`

It reads only the GGUF header (a few MiB per shard, by HTTP Range requests for a URL), so a 100 GB model costs
nothing to ask about. It applies the engine's own architecture guard, the geometry the kernels are built for, the
expert tensor types the engine reads, and the shapes and tensors of a whole split model.

```sh
python tools/model_check.py owner/repo                       # every model GGUF of a Hugging Face repo
python tools/model_check.py https://huggingface.co/owner/repo --match IQ3_XXS
python tools/model_check.py /models/Model-IQ3_XXS-00001-of-00002.gguf
python tools/model_check.py owner/repo-a owner/repo-b --json # several at once, machine-readable
```

Use the setup's `.venv` Python (`.venv/bin/python` / `.venv\Scripts\python`) to also get the packing hint, which needs
numpy. A gated repo needs `HF_TOKEN`; it is sent to the Hub only, never to the CDN it redirects to. `-v` logs what is
read.

| Verdict | Meaning |
| --- | --- |
| **OK** | `qwen4exp`, the geometry matches, every expert type is one the engine reads: it should run. |
| **CHECK** | It should run with a condition the report names: the experimental K-quant route ([UD-Q4_K_XL](UNSLOTH_Q4.md), [Q4_K_S](ORCA_Q4_K_S.md)), an unusual expert count, or shards that were not found. |
| **NO** | The report names why: another architecture, a geometry the kernels are not built for, an expert type the kernels do not read (Q5_0 down matrices), or missing tensors. |

Exit status: 0 for OK/CHECK only, 1 when a file is a NO, 2 when a file could not be read.

## What an OK does not tell you

It is a statement about the file's layout. Speed and quality of a fine-tune are not known until it is measured on
your PC: its draft head is the original model's (so token acceptance may differ), and `--compat-bf16` rounds the
projections the GGUF quantized. Measure it as [Community benchmark results](COMMUNITY_BENCHMARKS.md) describes.
