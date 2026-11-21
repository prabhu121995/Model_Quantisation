"""Benchmark every NF4-dq model in ./models: size on disk, load time, speed, and answer.

Run:  uv run python benchmarks/benchmark.py
      uv run python benchmarks/benchmark.py --prompt "What is NF4?" --max-new-tokens 50
Run nothing else heavy at the same time: models compete for CPU and RAM.
"""

import argparse
import gc
import time
from pathlib import Path

import psutil

from model_quantization.config import MODELS_DIR
from model_quantization.engine import Engine


def folder_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and f.suffix != ".md") / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--prompt", default="What is model quantisation? Answer in 2 short sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    dirs = sorted(d for d in args.models_dir.iterdir() if (d / "config.json").exists())
    if not dirs:
        raise SystemExit(f"No models in {args.models_dir}. Run `uv run model-quantization quantize` first.")

    me = psutil.Process()
    rows = []
    for d in dirs:
        print(f"Testing {d.name} ...")
        t = time.perf_counter()
        engine = Engine(d)
        load_s = time.perf_counter() - t
        ram = me.memory_info().rss / 1024**3
        engine.generate("Hi", max_new_tokens=2)  # warm-up (first call is slower)
        g = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
        rows.append((d.name, folder_gb(d), ram, load_s, g.tokens_per_second, g.text.replace("\n", " ")[:50]))
        del engine
        gc.collect()

    print(f"\nPrompt: {args.prompt}\n")
    print(f"{'model':<22}{'disk GB':>8}{'RAM GB':>8}{'load s':>8}{'tok/s':>7}  answer")
    for name, gb, ram, load_s, tps, text in rows:
        print(f"{name:<22}{gb:>8.2f}{ram:>8.2f}{load_s:>8.1f}{tps:>7.2f}  {text}")


if __name__ == "__main__":
    main()
