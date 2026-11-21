"""FastAPI server for the NF4 + double-quantised model.

Run:  uv run model-quantization serve
Docs: http://127.0.0.1:8000/docs
"""

import os
import threading
from typing import Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, engine

# `serve --model-dir` sets MODEL_DIR in the environment; otherwise use .env / defaults.
MODEL_DIR = os.environ.get("MODEL_DIR") or str(config.MODEL_DIR)

state: dict = {}
# The model is not thread-safe and CPU generation saturates all cores anyway,
# so requests are handled one at a time.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading NF4-dq model from {MODEL_DIR} ...")
    state["engine"] = engine.Engine(MODEL_DIR)
    print("Model ready.")
    yield
    state.pop("engine").close()


app = FastAPI(title="Qwen NF4 + Double Quantisation API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["What is model quantisation?"])
    max_new_tokens: int = Field(config.MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    thinking: bool = False
    kv_cache: Literal["paged", "dynamic", "none"] | None = None  # default from .env (KV_CACHE)


class GenerateResponse(BaseModel):
    text: str
    new_tokens: int
    seconds: float
    tokens_per_second: float
    prompt_tokens: int
    ttft_seconds: float  # time to first token (= prefill)
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    kv_cache: str
    kv_cache_blocks: int
    kv_cache_mb: float


@app.get("/health")
def health():
    eng = state.get("engine")
    info = {"status": "ok" if eng else "loading", "model_dir": MODEL_DIR}
    if eng:
        pool = eng.pool
        info["kv_pool"] = {"blocks": pool.num_blocks, "used_blocks": pool.used_blocks,
                           "block_size": pool.block_size, "total_mb": round(pool.total_mb, 1)}
    return info


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
            **({"kv_cache": req.kv_cache} if req.kv_cache else {}),
        )
    return GenerateResponse(
        text=g.text,
        new_tokens=g.new_tokens,
        seconds=round(g.seconds, 2),
        tokens_per_second=round(g.tokens_per_second, 2),
        prompt_tokens=g.prompt_tokens,
        ttft_seconds=round(g.ttft, 2),
        prefill_tokens_per_second=round(g.prefill_tps, 2),
        decode_tokens_per_second=round(g.decode_tps, 2),
        kv_cache=g.kv_cache,
        kv_cache_blocks=g.cache_blocks,
        kv_cache_mb=g.cache_mb,
    )
