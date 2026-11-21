"""FastAPI server for a quantised model (INT8 or GGUF backend).

Run:  uv run model-quantization serve                  # backend from .env (BACKEND=int8|gguf)
      uv run model-quantization serve --backend gguf
Docs: http://127.0.0.1:8000/docs
"""

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, engine

# `serve --backend/--model-path` set these in the environment; otherwise use .env / defaults.
BACKEND = os.environ.get("BACKEND") or config.BACKEND
MODEL_PATH = os.environ.get("MODEL_PATH") or str(engine.default_path(BACKEND))

state: dict = {}
# The model is not thread-safe and CPU generation saturates all cores anyway,
# so requests are handled one at a time.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {BACKEND} model from {MODEL_PATH} ...")
    state["engine"] = engine.get_engine(BACKEND, MODEL_PATH)
    print("Model ready.")
    yield
    state.pop("engine").close()


app = FastAPI(title="Quantised Qwen API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["What is model quantisation?"])
    max_new_tokens: int = Field(config.MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    thinking: bool = False


class GenerateResponse(BaseModel):
    text: str
    new_tokens: int
    seconds: float
    tokens_per_second: float


@app.get("/health")
def health():
    return {"status": "ok" if "engine" in state else "loading", "backend": BACKEND, "model_path": MODEL_PATH}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):  # sync def -> FastAPI runs it in a worker thread
    if "engine" not in state:
        raise HTTPException(503, "Model not loaded yet")
    with lock:
        g = state["engine"].generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            thinking=req.thinking,
        )
    return GenerateResponse(
        text=g.text,
        new_tokens=g.new_tokens,
        seconds=round(g.seconds, 2),
        tokens_per_second=round(g.tokens_per_second, 2),
    )
