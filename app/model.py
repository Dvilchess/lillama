import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.config import config
import logging

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        self.model = None

    def load(self):
        model_path = config.FINETUNED_MODEL_DIR if config.use_finetuned else config.MODEL_ID
        logger.info(f"Loading model from: {model_path} | ENV: {config.ENV}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info("Model loaded successfully")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipe(
            formatted,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
        )
        generated = outputs[0]["generated_text"]
        # Strip the prompt prefix
        return generated[len(formatted):].strip()

llm = LLMModel()
