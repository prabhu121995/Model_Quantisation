"""API tests with a fake engine, so they run in seconds."""

import pytest
from fastapi.testclient import TestClient

from model_quantization import api, engine
from model_quantization.engine import Generation


class FakeEngine:
    path = "fake"

    def generate(self, prompt, **kw):
        return Generation(text=f"echo: {prompt}", new_tokens=4, seconds=2.0)

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(engine, "get_engine", lambda backend, path: FakeEngine())
    with TestClient(api.app) as c:  # `with` runs the lifespan (model loading)
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] in engine.BACKENDS


def test_generate(client):
    r = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "echo: hi"
    assert body["tokens_per_second"] == 2.0


@pytest.mark.parametrize(
    "payload",
    [{"prompt": ""}, {"prompt": "hi", "max_new_tokens": 0}, {"prompt": "hi", "temperature": 5}],
)
def test_generate_rejects_bad_input(client, payload):
    assert client.post("/generate", json=payload).status_code == 422
