# Model Quantisation — Qwen3.5-4B with NF4 + Double Quantisation

Quantise **Qwen3.5-4B** with **bitsandbytes NF4 + double quantisation**, run it locally with an explicit
**prefill / decode** loop and a **paged KV cache** (the memory layout behind vLLM's PagedAttention), and
serve it through **FastAPI**. Built and tested on a CPU-only Windows laptop.

## Project layout

```
src/model_quantization/      # the application code
  config.py                  # settings, read from .env
  quantize.py                # download -> NF4 + double quantisation -> save
  inference.py               # load + generate: explicit PREFILL and DECODE phases
  kv_cache.py                # paged KV cache: block pool + block tables (PagedAttention layout)
  engine.py                  # model + tokenizer + KV block pool, used by CLI and API
  api.py                     # FastAPI app (/health, /generate)
  hub.py                     # upload to the Hugging Face Hub with a model card
  processes.py               # `status` / `stop` commands
  cli.py                     # `model-quantization quantize | run | serve | upload | status | stop`
notebooks/
  01_model_quantisation_guide.ipynb   # theory + NF4/double-quant + KV cache / prefill / decode
tests/                       # pytest (fast by default; `-m slow` for the real-model test)
benchmarks/benchmark.py      # disk size / RAM / load time / speed for every model in models/
scripts/                     # quantize_qwen.py, run_quantized.py shortcuts
models/                      # quantised models are saved here (git-ignored)
```

## How to run

```bash
uv sync                                   # 1. install (once)
cp .env.example .env                      #    settings (once; already done on this machine)

uv run model-quantization quantize        # 2. download Qwen3.5-4B (~9 GB) + quantise -> models/qwen3.5-4b-nf4-dq
                                          #    smaller test model: --model Qwen/Qwen3.5-0.8B

uv run model-quantization run "What is NF4?"   # 3. ask a question
uv run model-quantization serve                # 4. API -> http://127.0.0.1:8000/docs

uv run model-quantization status          # what is running, and how much RAM it uses
uv run model-quantization stop            # stop it
```

`run` prints both phases and the KV cache usage (format; your numbers will differ):

```
Prefill : <N> prompt tokens in <s>s = <x> tok/s  (time to first token)
Decode  : <N> tokens in <s>s = <x> tok/s
KV cache: <blocks> blocks x 16 tokens = <MB> MB
```

Compare cache modes: `--kv-cache paged` (default), `--kv-cache dynamic` (transformers' cache), or
`--kv-cache none` (no cache: every step re-processes the whole sequence, so it gets slower as the answer grows).

Call the API:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is double quantisation?", "max_new_tokens": 150, "kv_cache": "paged"}'
```

The response includes `ttft_seconds`, `prefill_tokens_per_second`, `decode_tokens_per_second`,
`kv_cache_blocks` and `kv_cache_mb`. `GET /health` shows the KV block pool (total / used blocks).

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-4B` | Hugging Face model to quantise |
| `MODELS_DIR` | `models` | Where quantised models are saved |
| `MODEL_DIR` | `models/qwen3.5-4b-nf4-dq` | Model loaded by `run` and `serve` |
| `KV_CACHE` | `paged` | `paged`, `dynamic` or `none` |
| `KV_BLOCK_SIZE` | `16` | Tokens per KV block |
| `KV_CACHE_TOKENS` | `4096` | KV pool capacity (prompt + answer); 4096 = 256 MB for the 4B model |
| `TORCH_THREADS` | `0` | CPU threads (0 = PyTorch default) |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | API address (`0.0.0.0` = reachable from your network) |
| `MAX_NEW_TOKENS` | `200` | Default answer length |
| `HF_USERNAME` | `PrabhuAIML` | Hugging Face account for `upload` |

Relative paths are resolved from the project root. Command-line options override `.env`.

## How it works

**NF4 + double quantisation.** Weights are stored as 4-bit NormalFloat in blocks of 64, each block with a
scale factor; double quantisation stores those scale factors in 8-bit too: ~4.127 instead of ~4.5 bits per
weight. Compute runs in float32 on CPU (most laptop CPUs have no fast bfloat16 math; bf16 was ~5x slower).

**Prefill and decode.** `inference.generate` runs the prompt through the model once (prefill: fills the KV
cache, gives the first token, compute-bound) and then one token per step (decode: memory-bound).

**Paged KV cache.** `kv_cache.py` pre-allocates one pool of KV blocks. Each request gets a block table
(logical -> physical block); blocks are allocated only when a block boundary is crossed and are returned to
the pool when the request ends, so at most one partly-filled block is wasted. Qwen3.5 is a hybrid model: only
its 8 full-attention layers (of 32) have a KV cache, 64 KB per token in float32; the linear-attention layers
keep a small fixed-size state. The paged cache gives exactly the same tokens as transformers' cache and as no
cache (see `tests/test_kv_cache.py`).

> A real PagedAttention *kernel* reads K/V directly from the blocks. Here the blocks are gathered into one
> tensor for PyTorch's attention, which is fine for one request at a time on CPU.

## Study & testing

```bash
# notebook: open notebooks/01_model_quantisation_guide.ipynb in VS Code, pick the .venv kernel
uv run pytest                                   # fast tests: config, API, double quant, paged KV cache, hub
uv run pytest -m slow                           # real quantise -> reload -> generate on Qwen3.5-0.8B
uv run python benchmarks/benchmark.py           # benchmark all models in models/
```

## Upload to Hugging Face

```bash
uv run hf auth login                            # once, with a token that has Write access
uv run model-quantization upload                # -> huggingface.co/PrabhuAIML/Qwen3.5-4B-NF4-DQ
```

## Notes

- Use the Hugging Face id `Qwen/Qwen3.5-4B`, **not** the Ollama tag `qwen3.5:4b`.
- On CPU, NF4 saves **memory**, not time: bitsandbytes de-quantises the 4-bit weights for each matrix
  multiply, so generation is slow (estimated well under 1 token/s for the 4B model on a laptop CPU, based on the 0.8B test).
