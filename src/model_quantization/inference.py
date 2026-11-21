"""INT8 backend: load a saved optimum-quanto model and generate text with PyTorch."""

import time
from pathlib import Path

import torch
from optimum.quanto import QuantizedModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# Hide "falling back to its reference PyTorch implementation because `causal_conv1d` /
# `flash-linear-attention` is not installed" - those kernels are GPU-only; the PyTorch
# fallback is correct on CPU, just slower. Real errors are still shown.
hf_logging.set_verbosity_error()

from .engine import Generation  # noqa: E402  (re-exported for notebooks / old imports)


def load(model_dir: str | Path):
    """Load an INT8 model saved by `quantize`."""
    model_dir = Path(model_dir)
    if not (model_dir / "quanto_qmap.json").exists():
        raise FileNotFoundError(
            f"No INT8 model in {model_dir}. Run `uv run model-quantization quantize` first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = QuantizedModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    thinking: bool = False,
) -> Generation:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,  # Qwen3.x "thinking" mode is slow on CPU; off by default
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
    seconds = time.perf_counter() - start

    new_ids = output[0][inputs["input_ids"].shape[1]:]
    return Generation(
        text=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        new_tokens=len(new_ids),
        seconds=seconds,
    )
