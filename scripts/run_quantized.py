"""Shortcut for:  uv run model-quantization run "your question"

e.g.:  uv run scripts/run_quantized.py "What is INT8?" --model-dir models/qwen3.5-0.8b-int8
"""
import sys

from model_quantization.cli import main

sys.argv[1:1] = ["run"]
main()
