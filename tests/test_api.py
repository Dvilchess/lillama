"""Tests básicos — se corren sin GPU usando mocks."""
import pytest
from unittest.mock import MagicMock, patch

def test_config_dev():
    import os
    os.environ["ENV"] = "dev"
    from app.config import Config
    cfg = Config()
    assert cfg.ENV == "dev"
    assert not cfg.is_prod

def test_config_prod():
    import os
    os.environ["ENV"] = "prod"
    from app.config import Config
    cfg = Config()
    assert cfg.is_prod

@patch("app.model.AutoModelForCausalLM.from_pretrained")
@patch("app.model.AutoTokenizer.from_pretrained")
def test_model_generate(mock_tok, mock_model):
    """Verifica que generate llame al pipeline correctamente."""
    from app.model import LLMModel
    m = LLMModel()
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "Hello world response"}]
    m.pipe = mock_pipe
    mock_tok_instance = MagicMock()
    mock_tok_instance.apply_chat_template.return_value = "Hello world"
    m.tokenizer = mock_tok_instance

    result = m.generate("Hello world", "You are helpful")
    assert isinstance(result, str)
