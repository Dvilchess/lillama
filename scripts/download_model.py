"""
Script para descargar TinyLlama (o cualquier modelo HF) localmente.
Uso: python scripts/download_model.py
"""
from huggingface_hub import snapshot_download
import os

MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

if __name__ == "__main__":
    print(f"Downloading {MODEL_ID} → {MODEL_DIR}")
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
    print("Done!")
