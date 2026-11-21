"""Benchmark every model in ./models (INT8 folders and GGUF folders): size, speed, and answer.

Run:  uv run python benchmarks/benchmark.py
      uv run python benchmarks/benchmark.py --filter 4b --max-new-tokens 50
Run nothing else heavy at the same time: models compete for CPU and RAM.
"""

import argparse
import gc
from pathlib import Path

from model_quantization.config import MODELS_DIR
from model_quantization.engine import get_engine


def folder_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and f.suffix != ".md") / 1024**3


def backend_of(d: Path) -> str | None:
    if (d / "quanto_qmap.json").exists():
        return "int8"
    if any(d.glob("*.gguf")):
        return "gguf"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--filter", default="", help="only folders whose name contains this, e.g. 4b")
    ap.add_argument("--prompt", default="What is model quantisation? Answer in 2 short sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    dirs = sorted(d for d in args.models_dir.iterdir() if d.is_dir() and backend_of(d) and args.filter in d.name)
    if not dirs:
        raise SystemExit(f"No models in {args.models_dir}.")

    rows = []
    for d in dirs:
        backend = backend_of(d)
        print(f"Testing {d.name} ({backend}) ...")
        engine = get_engine(backend, d)
        engine.generate("Hi", max_new_tokens=2)  # warm-up (first call is slower)
        g = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
        engine.close()
        rows.append((d.name, backend, folder_gb(d), g.tokens_per_second, g.text.replace("\n", " ")[:60]))
        del engine
        gc.collect()

    print(f"\nPrompt: {args.prompt}\n")
    print(f"{'model':<22}{'backend':<9}{'size GB':>8}{'tok/s':>8}  answer")
    for name, backend, gb, tps, text in rows:
        print(f"{name:<22}{backend:<9}{gb:>8.2f}{tps:>8.2f}  {text}")


if __name__ == "__main__":
    main()
