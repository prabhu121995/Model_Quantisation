"""GGUF backend: convert a Hugging Face model to GGUF, quantise it, and run it with llama.cpp.

llama.cpp has hand-optimised CPU kernels, so it is much faster on a laptop than PyTorch.
We use the official prebuilt Windows binaries (tools/llama.cpp/bin) and the converter
script from the matching source release (tools/llama.cpp/src). See README.
"""

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from . import config
from .engine import Generation


def _exe(name: str) -> Path:
    path = config.LLAMA_CPP_BIN / f"{name}.exe" if os.name == "nt" else config.LLAMA_CPP_BIN / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download llama.cpp into {config.LLAMA_CPP_BIN} (see README).")
    return path


# --------------------------------------------------------------------------- convert + quantise

def convert(model_name: str, out_dir: Path, quant: str = "Q4_K_M", keep_f16: bool = False) -> Path:
    """HF model -> GGUF f16 -> GGUF <quant>. Returns the quantised .gguf path."""
    from huggingface_hub import snapshot_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = model_name.split("/")[-1].lower()
    f16 = out_dir / f"{short}-f16.gguf"
    final = out_dir / f"{short}-{quant}.gguf"

    if not f16.exists():
        print(f"Downloading {model_name} ...")
        src = snapshot_download(model_name, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
        print("Converting to GGUF f16 ...")
        converter = config.LLAMA_CPP_SRC / "convert_hf_to_gguf.py"
        env = {**os.environ, "PYTHONPATH": str(config.LLAMA_CPP_SRC / "gguf-py")}
        subprocess.run(
            [sys.executable, str(converter), src, "--outtype", "f16", "--outfile", str(f16)], check=True, env=env
        )

    print(f"Quantising to {quant} ...")
    subprocess.run([str(_exe("llama-quantize")), str(f16), str(final), quant], check=True)
    if not keep_f16:
        f16.unlink()
    print(f"Done: {final} ({final.stat().st_size / 1024**3:.2f} GB)")
    return final


# --------------------------------------------------------------------------- run

class GGUFEngine:
    """Starts `llama-server` in the background and talks to it over HTTP."""

    def __init__(self, gguf_path: str | Path, port: int | None = None, ctx: int = 4096):
        gguf_path = Path(gguf_path)
        if gguf_path.is_dir():  # a folder -> use the .gguf inside it
            files = sorted(gguf_path.glob("*.gguf"))
            if not files:
                raise FileNotFoundError(f"No .gguf file in {gguf_path}")
            gguf_path = files[0]
        if not gguf_path.exists():
            raise FileNotFoundError(f"{gguf_path} not found. Run `uv run model-quantization gguf` first.")

        self.path = gguf_path
        self.port = port or config.LLAMA_SERVER_PORT
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [
            str(_exe("llama-server")), "-m", str(gguf_path),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(ctx), "--jinja",
        ]
        if config.LLAMA_THREADS:
            cmd += ["-t", str(config.LLAMA_THREADS)]
        # llama-server logs a lot; write it to a file (an unread pipe would fill up and block it).
        self.log_path = config.PROJECT_ROOT / "llama-server.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        self._wait_ready()

    def _wait_ready(self, timeout: float = 300) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"llama-server exited (log: {self.log_path}):\n{tail}")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not start in time")

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.0, thinking: bool = False) -> Generation:
        start = time.perf_counter()
        r = httpx.post(
            f"{self.url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
            timeout=None,
        )
        r.raise_for_status()
        body = r.json()
        t = body.get("timings")
        # prompt processing + generation, the same span the INT8 backend measures
        seconds = (t["prompt_ms"] + t["predicted_ms"]) / 1000 if t else time.perf_counter() - start
        return Generation(
            text=body["choices"][0]["message"]["content"].strip(),
            new_tokens=body["usage"]["completion_tokens"],
            seconds=seconds,
        )

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        self._log.close()
