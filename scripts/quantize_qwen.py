"""Shortcut for:  uv run model-quantization quantize  (Qwen3.5-4B, INT8).

Pass extra options through, e.g.:  uv run scripts/quantize_qwen.py --model Qwen/Qwen3.5-0.8B
"""
import sys

from model_quantization.cli import main

sys.argv[1:1] = ["quantize"]
main()
