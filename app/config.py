import os
from dataclasses import dataclass

@dataclass
class Config:
    ENV: str = os.getenv("ENV", "dev")

    # Model
    MODEL_ID: str = os.getenv("MODEL_ID", "./models")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    FINETUNED_MODEL_DIR: str = os.getenv("FINETUNED_MODEL_DIR", "./models/finetuned")

    # Inference
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))

    # Gradio
    GRADIO_HOST: str = os.getenv("GRADIO_HOST", "0.0.0.0")
    GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    # Auth (prod only)
    API_KEY: str = os.getenv("API_KEY", "")

    @property
    def is_prod(self) -> bool:
        return self.ENV == "prod"

    @property
    def use_finetuned(self) -> bool:
        return self.is_prod and os.path.exists(self.FINETUNED_MODEL_DIR)

config = Config()
