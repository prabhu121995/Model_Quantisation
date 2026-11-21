"""Download a model, quantise it to NF4 with double quantisation, and save it locally.

NF4 (4-bit NormalFloat) stores each weight in 4 bits, with one scale factor per block of
64 weights. Double quantisation also quantises those scale factors (FP32 -> 8-bit), which
saves another ~0.37 bits per weight.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _folder_size_gb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024**3


def nf4_dq_config() -> BitsAndBytesConfig:
    # Most laptop CPUs have no fast bfloat16 math: float32 compute was ~5x faster in testing.
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # <- double quantisation
        bnb_4bit_compute_dtype=compute_dtype,
    )


def download(model_name: str) -> str:
    """Download the model files (resumable), retrying if the connection drops or stalls."""
    import time

    from huggingface_hub import snapshot_download

    from .config import DOWNLOAD_RETRIES

    patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return snapshot_download(model_name, allow_patterns=patterns)
        except Exception as e:  # network errors / timeouts: partial files are kept and resumed
            if attempt == DOWNLOAD_RETRIES:
                raise
            print(f"Download interrupted ({type(e).__name__}); retry {attempt}/{DOWNLOAD_RETRIES - 1} in 5 s ...")
            time.sleep(5)


def quantize(model_name: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    print(f"Downloading {model_name} (first run can take a while; it resumes if interrupted)...")
    local = download(model_name)
    tokenizer = AutoTokenizer.from_pretrained(local)

    print("Quantising to NF4 + double quantisation...")
    config = nf4_dq_config()
    # bitsandbytes quantises while loading, layer by layer, so peak RAM stays low.
    model = AutoModelForCausalLM.from_pretrained(
        local,
        quantization_config=config,
        dtype=config.bnb_4bit_compute_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.config.base_model = model_name  # recorded for the Hub model card

    print(f"Saving to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Done. Size on disk: {_folder_size_gb(output_dir):.2f} GB")
    return output_dir
