"""Upload an NF4 + double-quantised model folder to the Hugging Face Hub, with a model card."""

import json
from pathlib import Path

from huggingface_hub import HfApi


def repo_name_for(model_dir: Path) -> str:
    """models/qwen3.5-4b-nf4-dq -> Qwen3.5-4B-NF4-DQ"""
    name = Path(model_dir).name.removesuffix("-nf4-dq")
    base, _, size = name.rpartition("-")
    return f"{base.capitalize()}-{size.upper()}-NF4-DQ"


def _base_model(model_dir: Path) -> str:
    cfg_file = Path(model_dir) / "config.json"
    if cfg_file.exists():
        base = json.loads(cfg_file.read_text()).get("base_model")
        if base:
            return base
    return "Qwen/" + repo_name_for(model_dir).removesuffix("-NF4-DQ")


def _size_gb(folder: Path) -> float:
    return sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file() and f.name != "README.md") / 1024**3


def model_card(repo_id: str, base: str, size_gb: float) -> str:
    return f"""---
license: apache-2.0
base_model: {base}
library_name: transformers
pipeline_tag: text-generation
tags:
- bitsandbytes
- nf4
- 4-bit
- double-quantization
- quantized
---

# {repo_id.split("/")[-1]}

**NF4 + double quantisation of [{base}](https://huggingface.co/{base})**, made with
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).

- Linear-layer weights: **4-bit NormalFloat (NF4)**, one scale factor per block of 64 weights
- **Double quantisation**: those scale factors are themselves quantised to 8-bit
  (~4.127 instead of ~4.5 bits per weight)
- Embeddings and `lm_head`: unquantised
- Size on disk: **{size_gb:.2f} GB**
- Text-only (the vision tower of the original model is not included)

## Usage

```bash
pip install "transformers>=5.23" "bitsandbytes>=0.50" accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto")  # NF4 settings come from config.json

messages = [{{"role": "user", "content": "What is model quantisation?"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## License

Apache-2.0, same as the base model [{base}](https://huggingface.co/{base}).
All credit for the model goes to the Qwen team; this repo only contains a quantised copy.
"""


def upload(model_dir: Path, username: str, private: bool = False) -> str:
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"No model in {model_dir}. Run `quantize` first.")

    repo_id = f"{username}/{repo_name_for(model_dir)}"
    (model_dir / "README.md").write_text(model_card(repo_id, _base_model(model_dir), _size_gb(model_dir)), encoding="utf-8")

    api = HfApi()
    print(f"Creating repo {repo_id} ({'private' if private else 'public'}) ...")
    api.create_repo(repo_id, private=private, exist_ok=True)
    print(f"Uploading {model_dir} ({_size_gb(model_dir):.2f} GB) ...")
    api.upload_folder(repo_id=repo_id, folder_path=model_dir, commit_message="Upload NF4 + double-quantised model")
    url = f"https://huggingface.co/{repo_id}"
    print(f"Done: {url}")
    return url
