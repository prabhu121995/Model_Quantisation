"""Download a model, quantise its weights to INT8, and save it locally."""

import json
from pathlib import Path

import torch
from optimum.quanto import freeze, qint8, quantization_map, quantize as quanto_quantize
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def _folder_size_gb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024**3


def _quantize_int8(model) -> None:
    """Quantise every decoder layer to INT8, one layer at a time.

    Each layer is first converted to float32 so the model *computes* in float32:
    most laptop CPUs have no fast bfloat16 math (bf16 was ~4x slower in testing).
    Doing it layer by layer keeps peak RAM close to the bf16 size, because each
    int8 layer (1 byte/weight) is smaller than the bf16 layer it replaces.
    """
    for layer in model.get_decoder().layers:
        layer.to(torch.float32)
        quanto_quantize(layer, weights=qint8)
        freeze(layer)
    # Embeddings, final norm and lm_head stay unquantised (they matter most for quality).
    model.to(torch.float32)
    model.config.dtype = torch.float32


def _save(model, output_dir: Path) -> None:
    """Save in the layout `QuantizedModelForCausalLM.from_pretrained` expects.

    transformers v5 `save_pretrained` renames some checkpoint keys (e.g. Qwen3.5's
    `model.*` -> `model.language_model.*`), which optimum-quanto 0.2.x's loader does
    not understand -> weights reload as zeros. So we write the raw state dict instead.
    """
    state_dict = model.state_dict()
    if model.config.tie_word_embeddings:
        state_dict.pop("lm_head.weight", None)  # re-tied on load
    save_file({k: v.contiguous() for k, v in state_dict.items()}, output_dir / "model.safetensors")
    (output_dir / "quanto_qmap.json").write_text(json.dumps(quantization_map(model), indent=2))
    model.config.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)


def quantize(model_name: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    print(f"Downloading + loading {model_name} (first run can take a while)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load in bfloat16: a 4B model needs ~8 GB RAM instead of ~16 GB in float32.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    print("Quantising weights to INT8...")
    _quantize_int8(model)

    print(f"Saving to {output_dir} ...")
    model.config.base_model = model_name  # recorded for the Hub model card
    output_dir.mkdir(parents=True, exist_ok=True)
    _save(model, output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Done. Size on disk: {_folder_size_gb(output_dir):.2f} GB")
    return output_dir
