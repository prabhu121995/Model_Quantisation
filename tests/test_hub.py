"""Model-card / repo-name logic for uploads (no network)."""

from pathlib import Path

from model_quantization import hub


def test_repo_names():
    assert hub.repo_name_for(Path("models/qwen3.5-0.8b-int8")) == "Qwen3.5-0.8B-INT8"
    assert hub.repo_name_for(Path("models/qwen3.5-4b-gguf")) == "Qwen3.5-4B-GGUF"


def test_gguf_card_lists_files(tmp_path):
    d = tmp_path / "qwen3.5-4b-gguf"
    d.mkdir()
    (d / "qwen3.5-4b-Q4_K_M.gguf").write_bytes(b"0" * 1024)
    assert hub.model_kind(d) == "gguf"
    card = hub.gguf_card("PrabhuAIML/Qwen3.5-4B-GGUF", hub._base_model(d), sorted(d.glob("*.gguf")))
    assert "base_model: Qwen/Qwen3.5-4B" in card
    assert "qwen3.5-4b-Q4_K_M.gguf" in card
    assert "ollama run hf.co/PrabhuAIML/Qwen3.5-4B-GGUF:Q4_K_M" in card
