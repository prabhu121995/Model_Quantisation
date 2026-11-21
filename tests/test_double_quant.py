"""Checks the double-quantisation behaviour itself, on a random tensor (no model download)."""

import pytest
import torch

bnb_f = pytest.importorskip("bitsandbytes.functional")


def _stats_bytes(state) -> int:
    """Bytes used by the per-block scale factors (the 'quantisation constants')."""
    b = state.absmax.numel() * state.absmax.element_size()
    if state.nested:
        b += state.state2.absmax.numel() * 4
    return b


@pytest.fixture(scope="module")
def weight():
    torch.manual_seed(0)
    return torch.randn(1024, 1024, dtype=torch.float32)


@pytest.fixture(scope="module")
def states(weight):
    return {
        dq: bnb_f.quantize_4bit(weight, quant_type="nf4", compress_statistics=dq)
        for dq in (False, True)
    }


def test_double_quant_produces_nested_state(states):
    assert not states[False][1].nested
    assert states[True][1].nested


def test_double_quant_shrinks_scale_factors(states):
    # fp32 scales -> 8-bit scales: roughly 4x smaller
    assert _stats_bytes(states[True][1]) < _stats_bytes(states[False][1]) / 3


def test_double_quant_keeps_accuracy(weight, states):
    err = {dq: (bnb_f.dequantize_4bit(q, s) - weight).abs().mean().item() for dq, (q, s) in states.items()}
    # Quantising the scales should barely change the reconstruction error.
    assert err[True] == pytest.approx(err[False], rel=0.02)
