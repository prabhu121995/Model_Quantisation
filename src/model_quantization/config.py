"""Central settings, read from the `.env` file in the project root (see `.env.example`).

Real environment variables override values in `.env`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Windows without Developer Mode has no symlinks, so huggingface_hub copies files instead and
# prints a long warning that looks like an error. Caching still works; hide the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Downloads: the newer "xet" transfer sometimes hangs forever on a slow connection. Plain HTTP
# with a timeout fails fast instead, and `quantize` then retries, resuming the partial file.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")  # seconds without data before giving up
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "10"))


def _path(name: str, default: str) -> Path:
    """Relative paths in .env are resolved from the project root, not the current folder."""
    p = Path(os.getenv(name) or default)
    return p if p.is_absolute() else PROJECT_ROOT / p


# Hugging Face repo id (NOT the Ollama tag "qwen3.5:4b").
DEFAULT_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-4B")

# Where quantised models are written: <MODELS_DIR>/<model>-nf4-dq
MODELS_DIR = _path("MODELS_DIR", "models")

# API server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Hugging Face Hub account used by `upload`
HF_USERNAME = os.getenv("HF_USERNAME", "")

# Generation defaults
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))

# KV cache: "paged" (block pool, like vLLM's PagedAttention), "dynamic" (transformers default),
# or "none" (no cache - recomputes everything each step; only for comparison)
KV_CACHE = os.getenv("KV_CACHE", "paged")
KV_BLOCK_SIZE = int(os.getenv("KV_BLOCK_SIZE", "16"))  # tokens per block
KV_CACHE_TOKENS = int(os.getenv("KV_CACHE_TOKENS", "4096"))  # pool capacity (prompt + answer)

# Original (not NF4) Qwen3.5-4B, served locally by Ollama - used by the /ask comparison endpoint
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1800"))  # seconds

# Every /ask call is appended to this CSV (start/end time, duration, prompt, response)
LOG_CSV = _path("LOG_CSV", "logs/requests.csv")

# CPU threads for PyTorch (0 = PyTorch default, usually the number of physical cores)
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "0"))


def default_output_dir(model_name: str = DEFAULT_MODEL) -> Path:
    short = model_name.split("/")[-1].lower()
    return MODELS_DIR / f"{short}-nf4-dq"


# The model the `run` / `serve` commands load. Defaults to the NF4-dq folder of MODEL_NAME.
MODEL_DIR = _path("MODEL_DIR", str(default_output_dir()))
