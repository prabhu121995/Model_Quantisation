"""Shortcut for:  uv run model-quantization run "your question"

e.g.:  uv run scripts/run_quantized.py "What is NF4?" --kv-cache none
"""
import sys

from model_quantization.cli import main

sys.argv[1:1] = ["run"]
main()
