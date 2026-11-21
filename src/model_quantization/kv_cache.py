"""Paged KV cache (the memory layout behind vLLM's PagedAttention), for teaching and measuring.

Why a KV cache?
    Attention needs the keys (K) and values (V) of every previous token. Without a cache, each
    new token recomputes K/V for the whole sequence: O(n^2) work. With a cache, prefill computes
    them once for the prompt, and each decode step only adds the new token's K/V: O(n).

Why *paged*?
    A plain cache grows one big contiguous tensor per sequence (re-allocated / concatenated as it
    grows). PagedAttention instead pre-allocates one pool of fixed-size **blocks** (e.g. 16 tokens)
    and gives each sequence a **block table**: logical block i -> physical block id, like virtual
    memory pages. Benefits: no fragmentation, at most one partly-filled block of waste per
    sequence, memory is freed back to the pool the moment a request finishes, and many sequences
    can share one pool (that is what lets vLLM batch many users).

Qwen3.5 is a hybrid model: only its full-attention layers (8 of 32 in the 4B model) have a KV
cache. The other layers are linear attention (Gated DeltaNet) with a small fixed-size state, which
is kept by transformers' normal cache layer.

Note: a real PagedAttention *kernel* reads K/V straight out of the blocks. Here we gather the
blocks into a contiguous tensor for PyTorch's attention, which is fine for one sequence on CPU.
"""

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class KVCacheFullError(MemoryError):
    pass


class BlockPool:
    """One pre-allocated pool of KV blocks shared by all full-attention layers and all requests."""

    def __init__(self, num_layers: int, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
        shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))  # stack of free physical block ids

    @classmethod
    def for_model(cls, model, max_tokens: int, block_size: int = 16) -> "BlockPool":
        cfg = model.config.get_text_config(decoder=True)
        layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
        n_attn = sum(t == "full_attention" for t in layer_types)
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        num_blocks = -(-max_tokens // block_size)  # ceil
        return cls(n_attn, num_blocks, block_size, cfg.num_key_value_heads, head_dim, model.dtype, model.device)

    # -- allocation ---------------------------------------------------------------------------
    def allocate(self) -> int:
        if not self._free:
            raise KVCacheFullError(
                f"KV cache full ({self.num_blocks} blocks x {self.block_size} tokens). "
                "Increase KV_CACHE_TOKENS in .env or ask for a shorter answer."
            )
        return self._free.pop()

    def release(self, block_ids: list[int]) -> None:
        self._free.extend(reversed(block_ids))

    # -- stats ----------------------------------------------------------------------------------
    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def total_mb(self) -> float:
        return 2 * self.k.numel() * self.k.element_size() / 1024**2

    @property
    def bytes_per_token(self) -> int:
        layers, _, heads, _, dim = self.k.shape
        return 2 * layers * heads * dim * self.k.element_size()


class Sequence:
    """One request's block table: logical block i -> physical block id in the pool."""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_table: list[int] = []

    def ensure_capacity(self, num_tokens: int) -> None:
        while len(self.block_table) * self.pool.block_size < num_tokens:
            self.block_table.append(self.pool.allocate())

    def free(self) -> None:
        self.pool.release(self.block_table)
        self.block_table = []


class PagedLayer(DynamicLayer):
    """Drop-in replacement for transformers' DynamicLayer that stores K/V in pool blocks."""

    def __init__(self, seq: Sequence, pool_layer: int):
        super().__init__()
        self.seq, self.pool_layer, self.length = seq, pool_layer, 0

    def lazy_initialization(self, key_states, value_states) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if key_states.shape[0] != 1:
            raise NotImplementedError("This teaching PagedLayer handles one sequence (batch size 1).")

        pool, bs = self.seq.pool, self.seq.pool.block_size
        start, n = self.length, key_states.shape[-2]
        end = start + n
        self.seq.ensure_capacity(end)  # allocates new blocks only when a block boundary is crossed

        # Write the new tokens into their (block, offset) slots.
        pos = torch.arange(start, end)
        table = torch.tensor(self.seq.block_table)
        blocks, offsets = table[pos // bs], pos % bs
        pool.k[self.pool_layer, blocks, :, offsets, :] = key_states[0].transpose(0, 1).to(pool.k.dtype)
        pool.v[self.pool_layer, blocks, :, offsets, :] = value_states[0].transpose(0, 1).to(pool.v.dtype)
        self.length = end

        # Gather this sequence's blocks (in block-table order) into contiguous K/V for attention.
        used = table[: -(-end // bs)]
        k = pool.k[self.pool_layer, used].permute(1, 0, 2, 3).reshape(1, pool.k.shape[2], -1, pool.k.shape[-1])
        v = pool.v[self.pool_layer, used].permute(1, 0, 2, 3).reshape(1, pool.v.shape[2], -1, pool.v.shape[-1])
        return k[:, :, :end].to(key_states.dtype), v[:, :, :end].to(value_states.dtype)

    def get_seq_length(self) -> int:
        return self.length

    def reset(self) -> None:
        self.length = 0


@dataclass
class CacheStats:
    tokens: int
    blocks: int
    block_size: int
    kv_mb: float

    @property
    def wasted_slots(self) -> int:
        """Unused slots in the last, partly filled block (the only waste with paging)."""
        return self.blocks * self.block_size - self.tokens


class PagedCache(DynamicCache):
    """transformers Cache whose full-attention layers are paged; linear-attention layers unchanged."""

    def __init__(self, model, pool: BlockPool):
        super().__init__(config=model.config)
        self.seq = Sequence(pool)
        attn_idx = 0
        for i, layer in enumerate(self.layers):
            if type(layer) is DynamicLayer:  # a full-attention layer -> page it
                self.layers[i] = PagedLayer(self.seq, attn_idx)
                attn_idx += 1

    def stats(self) -> CacheStats:
        pool, tokens = self.seq.pool, self.get_seq_length()
        return CacheStats(tokens, len(self.seq.block_table), pool.block_size,
                          len(self.seq.block_table) * pool.block_size * pool.bytes_per_token / 1024**2)

    def free(self) -> None:
        """Give this request's blocks back to the pool."""
        self.seq.free()
