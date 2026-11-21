"""Load a saved NF4 + double-quantised model and generate text."""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.utils import logging as hf_logging

from . import config

# Hide "falling back to its reference PyTorch implementation because `causal_conv1d` /
# `flash-linear-attention` is not installed" - those kernels are GPU-only; the PyTorch
# fallback is correct on CPU, just slower. Real errors are still shown.
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()  # the long "Loading weights: 1%..." bar

if config.TORCH_THREADS:
    torch.set_num_threads(config.TORCH_THREADS)

from .engine import Generation  # noqa: E402  (re-exported for notebooks)


def load(model_dir: str | Path):
    """Load a model saved by `quantize`. The NF4 settings are read from its config.json."""
    model_dir = Path(model_dir)
    cfg = model_dir / "config.json"
    if not cfg.exists() or '"quantization_config"' not in cfg.read_text():
        raise FileNotFoundError(
            f"No NF4 model in {model_dir}. Run `uv run model-quantization quantize` first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    return model, tokenizer


KV_CACHE_MODES = ("paged", "dynamic", "none")


def _eos_ids(model, tokenizer) -> set[int]:
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def _pick(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1))


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    thinking: bool = False,
    kv_cache: str | None = None,
    pool=None,
) -> Generation:
    """Generate an answer in two phases:

    1. PREFILL - run the whole prompt through the model in ONE forward pass. This fills the KV
       cache with every prompt token's keys/values and gives the first new token. Compute-bound,
       so it processes many tokens per second.
    2. DECODE - generate the rest one token at a time. Each step feeds only the newest token and
       reads all earlier K/V from the cache. Memory-bound (every step reads all the weights),
       so it is much slower per token.

    kv_cache: "paged" (block pool, like vLLM's PagedAttention; needs `pool`),
              "dynamic" (transformers' default growing cache), or
              "none" (no cache: every decode step re-processes the whole sequence - slow, for comparison).
    """
    from .kv_cache import PagedCache

    kv_cache = kv_cache or config.KV_CACHE
    if kv_cache not in KV_CACHE_MODES:
        raise ValueError(f"kv_cache must be one of {KV_CACHE_MODES}")
    if kv_cache == "paged" and pool is None:
        raise ValueError("kv_cache='paged' needs a BlockPool (Engine creates one).")

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,  # Qwen3.x "thinking" mode is slow on CPU; off by default
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    eos = _eos_ids(model, tokenizer)

    cache = {"paged": lambda: PagedCache(model, pool), "dynamic": lambda: DynamicCache(config=model.config),
             "none": lambda: None}[kv_cache]()
    use_cache = cache is not None
    new_ids: list[int] = []
    try:
        # ---- 1. PREFILL: whole prompt, one forward pass -------------------------------------
        t0 = time.perf_counter()
        logits = model(input_ids, past_key_values=cache, use_cache=use_cache).logits
        new_ids.append(_pick(logits[0, -1], temperature))
        prefill_s = time.perf_counter() - t0

        # ---- 2. DECODE: one token per step --------------------------------------------------
        t1 = time.perf_counter()
        while len(new_ids) < max_new_tokens and new_ids[-1] not in eos:
            if use_cache:  # only the newest token; earlier K/V come from the cache
                step_input = torch.tensor([[new_ids[-1]]], device=model.device)
            else:  # no cache: the whole sequence again
                step_input = torch.cat([input_ids, torch.tensor([new_ids], device=model.device)], dim=1)
            logits = model(step_input, past_key_values=cache, use_cache=use_cache).logits
            new_ids.append(_pick(logits[0, -1], temperature))
        decode_s = time.perf_counter() - t1

        stats = cache.stats() if kv_cache == "paged" else None
    finally:
        if kv_cache == "paged":
            cache.free()  # return the blocks to the pool for the next request

    return Generation(
        text=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        new_tokens=len(new_ids),
        seconds=prefill_s + decode_s,
        prompt_tokens=input_ids.shape[1],
        prefill_seconds=prefill_s,
        decode_seconds=decode_s,
        kv_cache=kv_cache,
        cache_blocks=stats.blocks if stats else 0,
        cache_mb=round(stats.kv_mb, 2) if stats else 0.0,
    )
