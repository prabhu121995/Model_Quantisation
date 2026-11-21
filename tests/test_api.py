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
