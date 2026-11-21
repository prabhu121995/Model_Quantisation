"""Real quantise -> save -> reload -> generate, on the small Qwen3.5-0.8B.

Slow (downloads ~1.7 GB once, a few minutes on CPU). Run with:
    uv run pytest -m slow
"""

import pytest
import torch

from model_quantization import inference
from model_quantization.quantize import quantize

pytestmark = pytest.mark.slow


def test_quantize_reload_generate(tmp_path):
    out = quantize("Qwen/Qwen3.5-0.8B", tmp_path / "int8")
    model, tokenizer = inference.load(out)
    # Computes in float32 (fast on CPU), weights stored as int8.
    assert model._wrapped.dtype == torch.float32
    g = inference.generate(model, tokenizer, "Name one planet. One word.", max_new_tokens=10)
    # Regression: models used to reload with zeroed weights and print "!!!!".
    assert "!!!" not in g.text
    assert g.text.strip(), "model produced no text"
