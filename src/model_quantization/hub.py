"""Upload a quantised model folder (INT8 or GGUF) to the Hugging Face Hub, with a model card."""

import json
from pathlib import Path

from huggingface_hub import HfApi


def model_kind(model_dir: Path) -> str:
    if (model_dir / "quanto_qmap.json").exists():
        return "int8"
    if any(model_dir.glob("*.gguf")):
        return "gguf"
    raise FileNotFoundError(f"{model_dir} has neither an INT8 model nor a .gguf file.")


def repo_name_for(model_dir: Path) -> str:
    """models/qwen3.5-0.8b-int8 -> Qwen3.5-0.8B-INT8,  models/qwen3.5-4b-gguf -> Qwen3.5-4B-GGUF"""
    name = model_dir.name
    for suffix in ("-int8", "-gguf"):
        if name.endswith(suffix):
            kind, name = suffix[1:].upper(), name.removesuffix(suffix)
            break
    else:
        kind = model_kind(model_dir).upper()
    base, _, size = name.rpartition("-")
    return f"{base.capitalize()}-{size.upper()}-{kind}"


def _base_model(model_dir: Path) -> str:
    cfg_file = model_dir / "config.json"
    if cfg_file.exists():
        base = json.loads(cfg_file.read_text()).get("base_model")
        if base:
            return base
    # Older folders / GGUF folders: derive it from the folder name.
    return "Qwen/" + repo_name_for(model_dir).rsplit("-", 1)[0]


def _size_gb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file() and f.name != "README.md") / 1024**3


_LICENSE = """## License

Apache-2.0, same as the base model [{base}](https://huggingface.co/{base}).
All credit for the model goes to the Qwen team; this repo only contains a quantised copy.
"""


def int8_card(repo_id: str, base: str, size_gb: float) -> str:
    return f"""---
license: apache-2.0
base_model: {base}
library_name: transformers
pipeline_tag: text-generation
tags:
- optimum-quanto
- int8
- quantized
- cpu
---

# {repo_id.split("/")[-1]}

**INT8 weight-only quantisation of [{base}](https://huggingface.co/{base})**, made with
[optimum-quanto](https://github.com/huggingface/optimum-quanto) for running on **CPU**.

- Linear layers of every decoder block: **int8 weights** (per-row scales)
- Embeddings, final norm and `lm_head`: unquantised
- Activations / compute: **float32** (faster than bfloat16 on most laptop CPUs)
- Size on disk: **{size_gb:.2f} GB**
- Text-only (the vision tower of the original model is not included)

## Usage

```bash
pip install "transformers>=5.23" "optimum-quanto>=0.2.7" accelerate
```

```python
from optimum.quanto import QuantizedModelForCausalLM
from transformers import AutoTokenizer

repo = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = QuantizedModelForCausalLM.from_pretrained(repo)

messages = [{{"role": "user", "content": "What is model quantisation?"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

> Load it with `QuantizedModelForCausalLM` (not `AutoModelForCausalLM`): the weights are stored in
> optimum-quanto's format, together with `quanto_qmap.json`.

{_LICENSE.format(base=base)}"""


def gguf_card(repo_id: str, base: str, files: list[Path]) -> str:
    rows = "\n".join(f"| `{f.name}` | {f.stem.rsplit('-', 1)[-1]} | {f.stat().st_size / 1024**3:.2f} GB |" for f in files)
    quant = files[0].stem.rsplit("-", 1)[-1]
    return f"""---
license: apache-2.0
base_model: {base}
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- quantized
- cpu
---

# {repo_id.split("/")[-1]}

**GGUF quantisation of [{base}](https://huggingface.co/{base})** for fast local inference with
[llama.cpp](https://github.com/ggml-org/llama.cpp), Ollama, LM Studio and other GGUF runtimes.
Converted with llama.cpp's `convert_hf_to_gguf.py` and quantised with `llama-quantize`. Text-only.

| File | Quant | Size |
|---|---|---|
{rows}

## Usage

**llama.cpp**
```bash
llama-server -hf {repo_id}:{quant} --jinja
# OpenAI-compatible API on http://127.0.0.1:8080/v1
```

**Ollama**
```bash
ollama run hf.co/{repo_id}:{quant}
```

**Python (llama-cpp-python)**
```python
from llama_cpp import Llama

llm = Llama.from_pretrained(repo_id="{repo_id}", filename="{files[0].name}")
print(llm.create_chat_completion(messages=[{{"role": "user", "content": "Hello!"}}])["choices"][0]["message"]["content"])
```

{_LICENSE.format(base=base)}"""


def upload(model_dir: Path, username: str, private: bool = False) -> str:
    model_dir = Path(model_dir)
    kind = model_kind(model_dir)
    repo_id = f"{username}/{repo_name_for(model_dir)}"
    base = _base_model(model_dir)

    if kind == "int8":
        card = int8_card(repo_id, base, _size_gb(model_dir))
        ignore = None
    else:
        files = sorted(f for f in model_dir.glob("*.gguf") if "-f16" not in f.name)
        card = gguf_card(repo_id, base, files)
        ignore = ["*-f16.gguf"]  # the unquantised intermediate file is not uploaded
    (model_dir / "README.md").write_text(card, encoding="utf-8")

    api = HfApi()
    print(f"Creating repo {repo_id} ({'private' if private else 'public'}) ...")
    api.create_repo(repo_id, private=private, exist_ok=True)
    print(f"Uploading {model_dir} ({_size_gb(model_dir):.2f} GB) ...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        ignore_patterns=ignore,
        commit_message=f"Upload {kind.upper()} quantised model",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Done: {url}")
    return url
