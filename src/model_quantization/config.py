"""Central settings, read from the `.env` file in the project root (see `.env.example`).

Real environment variables override values in `.env`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    """Relative paths in .env are resolved from the project root, not the current folder."""
    p = Path(os.getenv(name) or default)
    return p if p.is_absolute() else PROJECT_ROOT / p


# Hugging Face repo id (NOT the Ollama tag "qwen3.5:4b").
DEFAULT_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-4B")

# Where quantised models are written: <MODELS_DIR>/<model>-int8
MODELS_DIR = _path("MODELS_DIR", "models")

# API server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Hugging Face Hub account used by `upload`
HF_USERNAME = os.getenv("HF_USERNAME", "")

# Generation defaults
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))


def default_output_dir(model_name: str = DEFAULT_MODEL) -> Path:
    short = model_name.split("/")[-1].lower()
    return MODELS_DIR / f"{short}-int8"


# The model the `run` / `serve` commands load. Defaults to the INT8 folder of MODEL_NAME.
MODEL_DIR = _path("MODEL_DIR", str(default_output_dir()))


# --- GGUF / llama.cpp backend --------------------------------------------------------

# Which backend `run` / `serve` use: "int8" (optimum-quanto, PyTorch) or "gguf" (llama.cpp, faster on CPU)
BACKEND = os.getenv("BACKEND", "int8")

# llama.cpp quant type: Q4_K_M (recommended), Q5_K_M, Q8_0, ...
GGUF_QUANT = os.getenv("GGUF_QUANT", "Q4_K_M")


def gguf_output_dir(model_name: str = DEFAULT_MODEL) -> Path:
    short = model_name.split("/")[-1].lower()
    return MODELS_DIR / f"{short}-gguf"


# The .gguf file (or a folder containing one) loaded by the gguf backend
GGUF_PATH = _path("GGUF_PATH", str(gguf_output_dir()))

# llama.cpp prebuilt binaries + source (for the converter script)
LLAMA_CPP_BIN = _path("LLAMA_CPP_BIN", "tools/llama.cpp/bin")
LLAMA_CPP_SRC = _path("LLAMA_CPP_SRC", "tools/llama.cpp/src")

# Internal port of the llama-server helper process, and CPU threads (0 = llama.cpp decides)
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8081"))
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", "0"))
