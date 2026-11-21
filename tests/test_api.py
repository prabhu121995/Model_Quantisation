"""API tests with a fake engine, so they run in seconds."""

import pytest
from fastapi.testclient import TestClient

from model_quantization import api, engine
from model_quantization.engine import Generation


class FakePool:
    num_blocks, used_blocks, block_size, total_mb = 256, 0, 16, 256.0


class FakeEngine:
    def __init__(self, path=None):
        self.path = path
        self.pool = FakePool()

    def generate(self, prompt, **kw):
        return Generation(text=f"echo: {prompt}", new_tokens=4, seconds=2.0, prompt_tokens=10,
                          prefill_seconds=0.5, decode_seconds=1.5, kv_cache=kw.get("kv_cache", "paged"))

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(engine, "Engine", FakeEngine)
    with TestClient(api.app) as c:  # `with` runs the lifespan (model loading)
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_generate(client):
    r = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "echo: hi"
    assert body["tokens_per_second"] == 2.0
    assert body["ttft_seconds"] == 0.5
    assert body["prefill_tokens_per_second"] == 20.0
    assert body["decode_tokens_per_second"] == 2.0  # 3 decoded tokens in 1.5 s
    assert body["kv_cache"] == "paged"


def test_generate_kv_cache_choice(client):
    assert client.post("/generate", json={"prompt": "hi", "kv_cache": "none"}).json()["kv_cache"] == "none"
    assert client.post("/generate", json={"prompt": "hi", "kv_cache": "bogus"}).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{"prompt": ""}, {"prompt": "hi", "max_new_tokens": 0}, {"prompt": "hi", "temperature": 5}],
)
def test_generate_rejects_bad_input(client, payload):
    assert client.post("/generate", json=payload).status_code == 422


# ---- /ask + CSV log ----------------------------------------------------------------------------

from model_quantization import config, ollama_client, request_log  # noqa: E402


@pytest.fixture
def ask_client(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOG_CSV", tmp_path / "requests.csv")
    monkeypatch.setattr(ollama_client, "chat", lambda prompt, **kw: Generation(
        text=f"ollama: {prompt}", new_tokens=8, seconds=1.0, prompt_tokens=10))
    return client


def test_ask_both_logs_two_rows(ask_client):
    body = ask_client.post("/ask", json={"prompt": "hi\nthere, \"quoted\""}).json()
    assert [r["model"] for r in body["results"]] == ["ollama", "nf4-dq"]
    assert body["results"][0]["response"] == 'ollama: hi\nthere, "quoted"'
    assert body["results"][1]["response"].startswith("echo: hi")
    for r in body["results"]:
        assert r["status"] == "ok" and r["duration_sec"] >= 0
        assert r["duration_min"] == round(r["duration_sec"] / 60, 3)

    rows = request_log.read()
    assert len(rows) == 2 and {r["request_id"] for r in rows} == {body["request_id"]}
    assert rows[0]["model"] == "nf4-dq"  # newest first
    assert rows[1]["prompt"] == 'hi\nthere, "quoted"'  # newlines/quotes survive the CSV
    assert list(rows[0]) == request_log.COLUMNS

    logged = ask_client.get("/logs").json()["rows"]
    assert len(logged) == 2
    assert ask_client.get("/logs.csv").status_code == 200


def test_ask_single_model(ask_client):
    body = ask_client.post("/ask", json={"prompt": "hi", "model": "ollama"}).json()
    assert [r["model"] for r in body["results"]] == ["ollama"]
    assert len(request_log.read()) == 1


def test_ask_ollama_down_is_logged_and_other_model_still_runs(ask_client, monkeypatch):
    def down(prompt, **kw):
        raise ollama_client.OllamaError("Cannot reach Ollama")
    monkeypatch.setattr(ollama_client, "chat", down)
    res = ask_client.post("/ask", json={"prompt": "hi"}).json()["results"]
    assert res[0]["status"] == "error" and "Cannot reach Ollama" in res[0]["error"]
    assert res[1]["status"] == "ok"
    assert [r["status"] for r in request_log.read()] == ["ok", "error"]


def test_ask_rejects_unknown_model(ask_client):
    assert ask_client.post("/ask", json={"prompt": "hi", "model": "gpt"}).status_code == 422
