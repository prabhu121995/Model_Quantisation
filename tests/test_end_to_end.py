"""Real quantise -> save -> reload -> generate, on the small Qwen3.5-0.8B.

Slow (downloads ~1.7 GB once, a few minutes on CPU). Run with:
    uv run pytest -m slow
"""

import json

import pytest

from model_quantization import inference
from model_quantization.quantize import quantize

pytestmark = pytest.mark.slow


def test_quantize_reload_generate(tmp_path):
    out = quantize("Qwen/Qwen3.5-0.8B", tmp_path / "nf4-dq")
    qcfg = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qcfg["bnb_4bit_quant_type"] == "nf4"
    assert qcfg["bnb_4bit_use_double_quant"] is True

    model, tokenizer = inference.load(out)
    g = inference.generate(model, tokenizer, "Name one planet. One word.", max_new_tokens=10)
    assert g.text.strip(), "model produced no text"
