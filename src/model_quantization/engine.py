"""One interface over both backends, so the CLI and the API don't care which one runs.

    engine = get_engine("gguf")          # or "int8"
    g = engine.generate("Hello!")
"""

from dataclasses import dataclass
from pathlib import Path

from . import config

BACKENDS = ("int8", "gguf")


@dataclass
class Generation:
    text: str
    new_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.seconds if self.seconds else 0.0


class Int8Engine:
    """optimum-quanto INT8 model running in PyTorch."""

    def __init__(self, model_dir: str | Path):
        from . import inference  # imports torch; only needed for this backend

        self._inference = inference
        self.path = Path(model_dir)
        self.model, self.tokenizer = inference.load(model_dir)

    def generate(self, prompt: str, **kwargs) -> Generation:
        return self._inference.generate(self.model, self.tokenizer, prompt, **kwargs)

    def close(self) -> None:
        pass


def default_path(backend: str) -> Path:
    return config.GGUF_PATH if backend == "gguf" else config.MODEL_DIR


def get_engine(backend: str | None = None, path: str | Path | None = None):
    backend = backend or config.BACKEND
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {BACKENDS}.")
    path = path or default_path(backend)
    if backend == "gguf":
        from .gguf import GGUFEngine

        return GGUFEngine(path)
    return Int8Engine(path)
