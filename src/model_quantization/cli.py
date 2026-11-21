"""Command line entry point:  uv run model-quantization {quantize,run,serve,upload,status,stop} ..."""

import argparse
import os
import sys
from pathlib import Path

from . import config


def main() -> None:
    # Windows consoles default to cp1252, which can't print characters like "₄" that the model
    # may produce -> UnicodeEncodeError. Print UTF-8, replacing anything the console can't show.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="model-quantization")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("quantize", help="Download a model, quantise it to NF4 + double quantisation, and save it")
    q.add_argument("--model", default=config.DEFAULT_MODEL, help="Hugging Face repo id")
    q.add_argument("--output-dir", type=Path, help="Default: <MODELS_DIR>/<model>-nf4-dq")

    r = sub.add_parser("run", help="Ask the model a question")
    r.add_argument("prompt", nargs="?", default="What is model quantisation? Answer in 3 short sentences.")
    r.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    r.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    r.add_argument("--kv-cache", choices=("paged", "dynamic", "none"), default=config.KV_CACHE)

    s = sub.add_parser("serve", help="Start the FastAPI server")
    s.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    s.add_argument("--host", default=config.HOST)
    s.add_argument("--port", type=int, default=config.PORT)

    u = sub.add_parser("upload", help="Upload a quantised model to the Hugging Face Hub")
    u.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    u.add_argument("--username", default=config.HF_USERNAME, help="Hugging Face user or org")
    u.add_argument("--private", action="store_true")

    sub.add_parser("status", help="Show which models / servers of this project are running")
    sub.add_parser("stop", help="Stop running models, servers and benchmarks of this project")

    args = parser.parse_args()
    try:
        _dispatch(parser, args)
    except FileNotFoundError as e:  # e.g. model not quantised yet: show a short message, not a traceback
        raise SystemExit(f"Error: {e}") from None


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command in ("status", "stop"):
        from . import processes

        processes.status() if args.command == "status" else processes.stop()

    elif args.command == "quantize":
        from .quantize import quantize

        quantize(args.model, args.output_dir or config.default_output_dir(args.model))

    elif args.command == "run":
        from .engine import Engine

        print(f"Loading {args.model_dir} ...")
        engine = Engine(args.model_dir, kv_cache=args.kv_cache)
        print(f"Thinking... (KV cache: {args.kv_cache})")
        g = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
        print(f"\nINPUT : {args.prompt}\nOUTPUT: {g.text}\n")
        print(f"Prefill : {g.prompt_tokens} prompt tokens in {g.prefill_seconds:.2f}s = {g.prefill_tps:.1f} tok/s"
              "  (time to first token)")
        print(f"Decode  : {g.new_tokens - 1} tokens in {g.decode_seconds:.2f}s = {g.decode_tps:.2f} tok/s")
        if g.kv_cache == "paged":
            print(f"KV cache: {g.cache_blocks} blocks x {config.KV_BLOCK_SIZE} tokens = {g.cache_mb:.1f} MB")

    elif args.command == "upload":
        from .hub import upload

        if not args.username:
            parser.error("set HF_USERNAME in .env or pass --username")
        upload(args.model_dir, args.username, private=args.private)

    elif args.command == "serve":
        import uvicorn

        os.environ["MODEL_DIR"] = str(args.model_dir)
        uvicorn.run("model_quantization.api:app", host=args.host, port=args.port)
