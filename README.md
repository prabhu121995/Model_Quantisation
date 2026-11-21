# Model Quantisation — Qwen3.5-4B to INT8 on CPU

Quantise **Qwen3.5-4B** to **INT8** with optimum-quanto, run it locally, and serve it through a
**FastAPI** endpoint. Built and tested on a CPU-only Windows machine.

## Project layout

```
src/model_quantization/      # the application code
  config.py                  # default model and output paths
  quantize.py                # download -> quantise to INT8 -> save
  inference.py               # load a saved model + generate (reports tok/s)
  api.py                     # FastAPI app (/health, /generate)
  cli.py                     # `model-quantization quantize | run | serve`
notebooks/                   # study material
  01_model_quantisation_guide.ipynb
tests/                       # pytest (fast by default; `-m slow` for the real-model test)
benchmarks/
  benchmark.py               # size / speed / answer for every model in models/
scripts/
  quantize_qwen.py           # shortcut for `model-quantization quantize`
  run_quantized.py           # shortcut for `model-quantization run`
models/                      # quantised models are saved here (git-ignored)
```

## Configuration (`.env`)

Common settings live in `.env` in the project root (git-ignored). Copy the template once:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-4B` | Hugging Face model to quantise |
| `MODELS_DIR` | `models` | Where quantised models are saved |
| `MODEL_DIR` | `models/qwen3.5-4b-int8` | Model loaded by `run` and `serve` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | API server address (`0.0.0.0` = reachable from your network) |
| `MAX_NEW_TOKENS` | `200` | Default answer length |

Relative paths are resolved from the project root. Command-line options (e.g. `--model-dir`)
and real environment variables override `.env`.

## Quick start

```bash
uv sync

# 1. Quantise Qwen/Qwen3.5-4B to INT8 -> models/qwen3.5-4b-int8
uv run model-quantization quantize
#    smaller model for testing: --model Qwen/Qwen3.5-0.8B

# 2. Ask a question
uv run model-quantization run "What is model quantisation?"

# 3. Serve with FastAPI  ->  http://127.0.0.1:8000/docs
uv run model-quantization serve
```

Call the API:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is INT8 quantisation?", "max_new_tokens": 150}'
```

## Study & testing

```bash
# study notebook: open notebooks/*.ipynb in VS Code and pick the .venv kernel
uv run pytest                                   # fast tests (~30 s): config, API
uv run pytest -m slow                           # real quantise -> reload -> generate on Qwen3.5-0.8B
uv run python benchmarks/benchmark.py           # benchmark all models in models/
```

## How it works

- The model is loaded in bfloat16 (a 4B model needs ~8 GB RAM instead of ~16 GB in FP32).
- Each decoder layer is converted to float32 and quantised to INT8 **one layer at a time**, so RAM
  stays low. The finished model stores weights in int8 but **computes in float32**: most laptop CPUs
  have no fast bfloat16 math (bf16 was ~3–4x slower in testing).
- Embeddings and `lm_head` stay unquantised, since they affect quality the most.

## Benchmark (Qwen3.5-0.8B, Intel i5-8265U, CPU only)

| Setup | Size on disk | Speed |
|---|---|---|
| INT8, float32 compute (this project) | 1.43 GB | 0.49 tok/s |
| INT8, bfloat16 compute | 0.96 GB | 0.11–0.16 tok/s |

Short runs (4–20 tokens), so treat them as rough. On CPU, quanto INT8 mainly saves **memory**
(it is what lets the 4B model fit in 16 GB RAM); it is not faster than the unquantised float32 model.
For maximum CPU speed, see GGUF / llama.cpp in the notebook's bonus section.

## Notes

- Use the Hugging Face id `Qwen/Qwen3.5-4B`, **not** the Ollama tag `qwen3.5:4b`.
- Models are saved with raw state-dict keys, because transformers v5 renames Qwen3.5 keys on
  `save_pretrained` in a way optimum-quanto 0.2.x cannot reload (weights came back as zeros).
