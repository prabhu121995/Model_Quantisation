"""Call the original Qwen3.5-4B running locally in Ollama (`ollama run qwen3.5:4b`).

Uses only the standard library. Returns the same `Generation` object as the NF4-dq engine,
so both models can be compared and logged the same way.
"""

import json
import urllib.error
import urllib.request

from . import config
from .engine import Generation


class OllamaError(RuntimeError):
    pass


def chat(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    thinking: bool = False,
    model: str | None = None,
    url: str | None = None,
    timeout: float | None = None,
) -> Generation:
    model = model or config.OLLAMA_MODEL
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": thinking,  # same as enable_thinking for the NF4 model
        "options": {"num_predict": max_new_tokens, "temperature": temperature},
    }
    req = urllib.request.Request(
        f"{(url or config.OLLAMA_URL).rstrip('/')}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or config.OLLAMA_TIMEOUT) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise OllamaError(f"Ollama returned {e.code}: {e.read().decode(errors='replace')[:300]}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise OllamaError(f"Cannot reach Ollama at {config.OLLAMA_URL} ({e}). Is `ollama serve` running?") from None

    ns = 1e9  # Ollama reports durations in nanoseconds
    prefill = data.get("prompt_eval_duration", 0) / ns
    decode = data.get("eval_duration", 0) / ns
    return Generation(
        text=data.get("message", {}).get("content", "").strip(),
        new_tokens=data.get("eval_count", 0),
        seconds=prefill + decode,
        prompt_tokens=data.get("prompt_eval_count", 0),
        prefill_seconds=prefill,
        decode_seconds=decode,
        kv_cache="ollama",
    )
