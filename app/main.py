import gradio as gr
from app.model import llm
from app.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def respond(message: str, history: list, system_prompt: str) -> str:
    return llm.generate(message, system_prompt)

def build_ui() -> gr.Blocks:
    env_badge = "🟢 DEV" if not config.is_prod else "🔴 PROD"
    model_label = "finetuned" if config.use_finetuned else config.MODEL_ID

    with gr.Blocks(title="TinyLlama Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# 🦙 TinyLlama Chat  `{env_badge}`")
        gr.Markdown(f"**Model:** `{model_label}`")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        gr.Textbox(
                            value="You are a helpful assistant.",
                            label="System Prompt",
                            lines=2,
                        )
                    ],
                    type="messages",
                )

    return demo

def main():
    logger.info(f"Starting app | ENV={config.ENV} | Model={config.MODEL_ID}")
    llm.load()
    demo = build_ui()
    demo.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=config.GRADIO_SHARE,
    )

if __name__ == "__main__":
    main()
