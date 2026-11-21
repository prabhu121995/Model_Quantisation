"""FastAPI server for the NF4 + double-quantised model.

Run:  uv run model-quantization serve
Docs: http://127.0.0.1:8000/docs
"""

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config, engine, ollama_client, request_log

# `serve --model-dir` sets MODEL_DIR in the environment; otherwise use .env / defaults.
MODEL_DIR = os.environ.get("MODEL_DIR") or str(config.MODEL_DIR)

state: dict = {}
# The model is not thread-safe and CPU generation saturates all cores anyway,
# so requests are handled one at a time. /ask also holds this lock while Ollama answers,
# so the two models never compete for the CPU and their timings are comparable.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading NF4-dq model from {MODEL_DIR} ...")
    state["engine"] = engine.Engine(MODEL_DIR)
    print("Model ready.")
    yield
    state.pop("engine").close()


app = FastAPI(title="Qwen NF4 + Double Quantisation API", lifespan=lifespan)

ModelChoice = Literal["ollama", "nf4-dq", "both"]


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
    info["ollama"] = {"url": config.OLLAMA_URL, "model": config.OLLAMA_MODEL, "reachable": _ollama_up()}
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


# ---- /ask: original (Ollama) vs NF4 + double-quantised, every call logged to CSV --------------

def _ollama_up() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL.rstrip('/')}/api/tags", timeout=2):
            return True
    except Exception:
        return False


class AskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["What is model quantisation? Answer in 3 sentences."])
    model: ModelChoice = "both"
    max_new_tokens: int = Field(config.MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    thinking: bool = False


class AskResult(BaseModel):
    model: str  # "ollama" or "nf4-dq"
    model_name: str
    start_time: str
    end_time: str
    duration_sec: float
    duration_min: float
    response: str
    prompt_tokens: int = 0
    new_tokens: int = 0
    tokens_per_second: float = 0.0
    status: Literal["ok", "error"] = "ok"
    error: str = ""


class AskResponse(BaseModel):
    request_id: str
    prompt: str
    results: list[AskResult]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _run_one(which: str, req: AskRequest):
    opts = dict(max_new_tokens=req.max_new_tokens, temperature=req.temperature, thinking=req.thinking)
    if which == "ollama":
        return config.OLLAMA_MODEL, lambda: ollama_client.chat(req.prompt, **opts)
    return MODEL_DIR, lambda: state["engine"].generate(req.prompt, **opts)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask the original model (Ollama), our NF4-dq model, or both (one after the other).

    Each model's answer is appended to the CSV log (LOG_CSV) with start/end time and duration.
    A failure of one model is returned and logged as status="error"; the other still runs.
    """
    if req.model in ("nf4-dq", "both") and "engine" not in state:
        raise HTTPException(503, "Model not loaded yet")
    request_id = uuid.uuid4().hex[:12]
    results = []
    for which in (["ollama", "nf4-dq"] if req.model == "both" else [req.model]):
        name, call = _run_one(which, req)
        with lock:
            start, t0 = _now(), time.perf_counter()
            try:
                g, err = call(), ""
            except Exception as e:  # noqa: BLE001 - log it and keep going with the other model
                g, err = None, f"{type(e).__name__}: {e}"
            secs, end = time.perf_counter() - t0, _now()
        r = AskResult(
            model=which, model_name=str(name), start_time=start, end_time=end,
            duration_sec=round(secs, 2), duration_min=round(secs / 60, 3),
            response=g.text if g else "",
            prompt_tokens=g.prompt_tokens if g else 0,
            new_tokens=g.new_tokens if g else 0,
            tokens_per_second=round(g.tokens_per_second, 2) if g else 0.0,
            status="ok" if g else "error", error=err,
        )
        request_log.append({"request_id": request_id, "prompt": req.prompt, **r.model_dump()})
        results.append(r)
    return AskResponse(request_id=request_id, prompt=req.prompt, results=results)


@app.get("/logs")
def logs(limit: int = Query(200, ge=1, le=100_000)):
    """The CSV log as JSON rows, newest first."""
    return {"path": str(config.LOG_CSV), "rows": request_log.read(limit=limit)}


@app.get("/logs.csv")
def logs_csv():
    """Download the raw CSV log."""
    if not config.LOG_CSV.exists():
        raise HTTPException(404, "No requests logged yet")
    return FileResponse(config.LOG_CSV, media_type="text/csv", filename=config.LOG_CSV.name)
