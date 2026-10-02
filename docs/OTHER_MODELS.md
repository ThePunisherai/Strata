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
  Those files belong in a general-purpose runtime such as llama.cpp - see
  [below](#not-qwen4exp-llamacpps-expert-offload-planned-for-your-pc) for the same idea there.

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

## Not qwen4exp: llama.cpp's expert offload, planned for your PC

For a file that is not `qwen4exp`, the idea behind Strata's speed - attention and the dense layers on the GPU, the
experts in RAM, as many layers' experts on the GPU as the VRAM leaves room for - exists in llama.cpp as `-ngl` plus
`--n-cpu-moe` (or `-ot` in older builds). `tools/llama_plan.py` reads the GGUF header, does the memory arithmetic for
**your** VRAM, RAM and context, and prints the flags; with `--run` it starts `llama-server` on Strata's address
(`127.0.0.1:8080`, API at `/v1`), so a client keeps its URL. It never starts Strata's engine: stop one before the other.

```sh
python tools/llama_plan.py /models/Model-Q4_K_M-00001-of-00002.gguf              # VRAM and RAM are detected
python tools/llama_plan.py owner/repo --match Q4_K_M --vram 16 --ram 32          # a repo: headers only, no download
python tools/llama_plan.py /models/Model-Q4_K_M.gguf --ctx 65536 --kv q8_0 --ram-bw 60 --run
```

```
  on the GPU   48 of 48 layers + output; experts of the last 29 layers (~14.3 GiB)
  in RAM       the rest (~7.1 GiB)
  flags        -c 32768 -ngl 999 --n-cpu-moe 19 --jinja --host 127.0.0.1 --port 8080
```

(An example from a synthetic 30B-class MoE header on 16 GiB of VRAM and 32 GiB of RAM; every number is arithmetic on
the header, not a measurement.)

- **Placement.** The dense layers and their KV cache go to the GPU from the last layer backwards; if everything fits,
  the remaining VRAM takes whole layers' experts, again from the last layer. The first `--n-cpu-moe N` layers keep
  theirs in RAM. A dense model gets only `-ngl`.
- **Verdict.** OK: what stays in RAM leaves `--reserve-ram` (8 GiB) for the system. TIGHT: it fits, barely. SLOW: it does
  not, and the rest is read from the SSD (mmap) while it answers. `--run` refuses a SLOW plan without `--force`, a port
  that is in use, and a repo target (download the file first).
- **KV cache.** From the attention keys of the header: K and V for `--ctx` tokens at `--kv` (f16, q8_0, q4_0). A hybrid
  model with a per-layer table or `full_attention_interval` counts its attention layers only; a recurrent model
  without one is an upper bound, and a header that does not say gets `--kv-gib`.
- **Builds.** `--run` reads `llama-server --help`: `--n-cpu-moe` or the `-ot` pattern, and flash attention (needed by a
  quantized cache) in the spelling that build uses. Without `--run` both spellings are printed.
- **`--ram-bw`** (GB/s) adds a roofline: the decode speed if the CPU's only job were reading the active experts. The
  real speed is lower, but it ranks models and quantizations against each other.

What it does not give you: Strata's speed. llama.cpp keeps no hot-expert cache on the GPU, does not overlap the CPU
and GPU experts per layer and has no MTP draft head, so the same model is slower than Strata would run it; and it runs
only what llama.cpp implements. Whether a given file loads there is llama.cpp's to say.
