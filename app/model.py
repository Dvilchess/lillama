import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
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

        if torch.cuda.is_available():
            logger.info("GPU detectada — cargando en 4-bit")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            logger.info("CPU detectada — cargando con low_cpu_mem_usage")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info("Modelo cargado!")

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
            return_full_text=False,
        )
        generated = outputs[0]["generated_text"]
        return generated.strip()
        

llm = LLMModel()