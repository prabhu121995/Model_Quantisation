"""A small wrapper that keeps a loaded model, its tokenizer and its KV block pool together.

    engine = Engine("models/qwen3.5-4b-nf4-dq")
    g = engine.generate("Hello!")
    print(g.text, g.ttft, g.decode_tps)
"""

from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Generation:
    text: str
    new_tokens: int
    seconds: float  # prefill + decode
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    kv_cache: str = ""
    cache_blocks: int = 0  # paged cache: blocks this request used
    cache_mb: float = 0.0  # paged cache: KV memory this request used

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.seconds if self.seconds else 0.0

    @property
    def ttft(self) -> float:
        """Time to first token = the prefill time."""
        return self.prefill_seconds

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tps(self) -> float:
        n = self.new_tokens - 1  # the first token comes out of prefill
        return n / self.decode_seconds if self.decode_seconds and n > 0 else 0.0


class Engine:
    """NF4 + double-quantised model with a paged KV cache."""

    def __init__(self, model_dir: str | Path | None = None, kv_cache: str | None = None):
        from . import inference  # imports torch; kept lazy so `--help` etc. stay fast
        from .kv_cache import BlockPool

        self._inference = inference
        self.path = Path(model_dir or config.MODEL_DIR)
        self.kv_cache = kv_cache or config.KV_CACHE
        self.model, self.tokenizer = inference.load(self.path)
        # Pre-allocate the KV block pool once (like vLLM); requests borrow blocks and return them.
        self.pool = BlockPool.for_model(self.model, config.KV_CACHE_TOKENS, config.KV_BLOCK_SIZE)

    def generate(self, prompt: str, **kwargs) -> Generation:
        kwargs.setdefault("kv_cache", self.kv_cache)
        return self._inference.generate(self.model, self.tokenizer, prompt, pool=self.pool, **kwargs)

    def close(self) -> None:
        pass
