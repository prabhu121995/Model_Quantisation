"""Model-card / repo-name logic for uploads (no network)."""

from pathlib import Path

from model_quantization import hub


def test_repo_name():
    assert hub.repo_name_for(Path("models/qwen3.5-4b-nf4-dq")) == "Qwen3.5-4B-NF4-DQ"


def test_model_card():
    card = hub.model_card("PrabhuAIML/Qwen3.5-4B-NF4-DQ", "Qwen/Qwen3.5-4B", 3.5)
    assert "base_model: Qwen/Qwen3.5-4B" in card
    assert "double-quantization" in card
    assert 'from_pretrained(repo, device_map="auto")' in card
