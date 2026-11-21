"""Command line entry point:  uv run model-quantization {quantize,gguf,run,serve,upload} ..."""

import argparse
import os
from pathlib import Path

from . import config
from .engine import BACKENDS


def main() -> None:
    parser = argparse.ArgumentParser(prog="model-quantization")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("quantize", help="Download a model, quantise it to INT8 (optimum-quanto), and save it")
    q.add_argument("--model", default=config.DEFAULT_MODEL, help="Hugging Face repo id")
    q.add_argument("--output-dir", type=Path, help="Default: <MODELS_DIR>/<model>-int8")

    g = sub.add_parser("gguf", help="Convert a model to GGUF and quantise it with llama.cpp")
    g.add_argument("--model", default=config.DEFAULT_MODEL, help="Hugging Face repo id")
    g.add_argument("--quant", default=config.GGUF_QUANT, help="Q4_K_M (default), Q5_K_M, Q8_0, ...")
    g.add_argument("--output-dir", type=Path, help="Default: <MODELS_DIR>/<model>-gguf")

    path_help = "INT8 model folder, or .gguf file/folder (default from .env)"

    r = sub.add_parser("run", help="Ask the model a question")
    r.add_argument("prompt", nargs="?", default="What is model quantisation? Answer in 3 short sentences.")
    r.add_argument("--backend", choices=BACKENDS, default=config.BACKEND)
    r.add_argument("--model-path", "--model-dir", type=Path, help=path_help)
    r.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)

    s = sub.add_parser("serve", help="Start the FastAPI server")
    s.add_argument("--backend", choices=BACKENDS, default=config.BACKEND)
    s.add_argument("--model-path", "--model-dir", type=Path, help=path_help)
    s.add_argument("--host", default=config.HOST)
    s.add_argument("--port", type=int, default=config.PORT)

    u = sub.add_parser("upload", help="Upload a quantised model (INT8 or GGUF folder) to the Hugging Face Hub")
    u.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    u.add_argument("--username", default=config.HF_USERNAME, help="Hugging Face user or org")
    u.add_argument("--private", action="store_true")

    args = parser.parse_args()

    if args.command == "quantize":
        from .quantize import quantize

        quantize(args.model, args.output_dir or config.default_output_dir(args.model))

    elif args.command == "gguf":
        from .gguf import convert

        convert(args.model, args.output_dir or config.gguf_output_dir(args.model), quant=args.quant)

    elif args.command == "run":
        from .engine import get_engine

        engine = get_engine(args.backend, args.model_path)
        print(f"[{args.backend}] {engine.path}\nThinking...")
        result = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
        engine.close()
        print(f"\nINPUT : {args.prompt}\nOUTPUT: {result.text}")
        print(f"\n[{result.new_tokens} tokens in {result.seconds:.1f}s = {result.tokens_per_second:.2f} tok/s]")

    elif args.command == "upload":
        from .hub import upload

        if not args.username:
            parser.error("set HF_USERNAME in .env or pass --username")
        upload(args.model_dir, args.username, private=args.private)

    elif args.command == "serve":
        import uvicorn

        os.environ["BACKEND"] = args.backend
        if args.model_path:
            os.environ["MODEL_PATH"] = str(args.model_path)
        uvicorn.run("model_quantization.api:app", host=args.host, port=args.port)
