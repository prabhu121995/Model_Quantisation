"""Paged KV cache correctness, on a tiny random Qwen3.5 model built in memory (no download).

The paged cache, transformers' DynamicCache and no cache at all must give the SAME tokens.
"""

import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from model_quantization.kv_cache import BlockPool, KVCacheFullError, PagedCache, PagedLayer


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
    )
    return Qwen3_5ForCausalLM(cfg).eval()


def greedy(model, prompt, steps, make_cache):
    """Prefill the prompt, then decode `steps` tokens one by one."""
    cache = make_cache()
    with torch.inference_mode():
        ids = [int(model(prompt, past_key_values=cache, use_cache=cache is not None).logits[0, -1].argmax())]
        for _ in range(steps - 1):
            if cache is None:
                x = torch.cat([prompt, torch.tensor([ids])], dim=1)
            else:
                x = torch.tensor([[ids[-1]]])
            ids.append(int(model(x, past_key_values=cache, use_cache=cache is not None).logits[0, -1].argmax()))
    return ids, cache


PROMPT = torch.tensor([[5, 17, 42, 99, 3, 8, 23, 77, 11, 64, 2, 31, 7, 90, 15, 44, 21, 19, 6]])  # 19 tokens


def test_paged_matches_dynamic_and_no_cache(model):
    pool = BlockPool.for_model(model, max_tokens=256, block_size=4)
    paged, cache = greedy(model, PROMPT, 12, lambda: PagedCache(model, pool))
    dynamic, _ = greedy(model, PROMPT, 12, lambda: DynamicCache(config=model.config))
    none, _ = greedy(model, PROMPT, 12, lambda: None)
    assert paged == dynamic == none

    with torch.inference_mode():  # and the same as transformers' own generate()
        ref = model.generate(PROMPT, max_new_tokens=12, do_sample=False, min_new_tokens=12)
    assert paged == ref[0, PROMPT.shape[1]:].tolist()

    # 19 prompt + 11 decoded tokens in the cache (the 12th is predicted, not yet stored)
    stats = cache.stats()
    assert stats.tokens == 30
    assert stats.blocks == 8  # ceil(30 / 4)
    assert stats.wasted_slots == 2  # only the last block is partly empty
    assert sum(isinstance(layer, PagedLayer) for layer in cache.layers) == 1  # 1 full-attention layer


def test_blocks_are_returned_to_the_pool(model):
    pool = BlockPool.for_model(model, max_tokens=64, block_size=8)
    _, cache = greedy(model, PROMPT, 5, lambda: PagedCache(model, pool))
    assert pool.used_blocks == 3  # 23 tokens -> 3 blocks of 8
    cache.free()
    assert pool.used_blocks == 0


def test_pool_full_raises(model):
    pool = BlockPool.for_model(model, max_tokens=8, block_size=4)  # room for 8 tokens only
    with pytest.raises(KVCacheFullError):
        greedy(model, PROMPT, 2, lambda: PagedCache(model, pool))
