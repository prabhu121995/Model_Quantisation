from model_quantization import config


def test_default_model_is_a_valid_hf_repo_id():
    # Regression: "Qwen/qwen3.5:4b" (Ollama tag) is rejected by Hugging Face.
    assert ":" not in config.DEFAULT_MODEL
    assert config.DEFAULT_MODEL.count("/") == 1


def test_default_output_dir():
    out = config.default_output_dir("Qwen/Qwen3.5-4B")
    assert out == config.MODELS_DIR / "qwen3.5-4b-nf4-dq"


def test_paths_are_absolute():
    # Relative paths from .env are resolved against the project root,
    # so the CLI, notebooks/ and tests/ all point at the same folder.
    assert config.MODELS_DIR.is_absolute()
    assert config.MODEL_DIR.is_absolute()
