# 🦙 TinyLlama Fine-tuning Project

Pipeline completo: carga del modelo → fine-tuning → deploy, con separación `dev / prod / feature/*`.

## Stack
- **Modelo base**: TinyLlama 1.1B Chat
- **Fine-tuning**: LoRA + PEFT en Google Colab
- **UI/API**: Gradio
- **Infra**: Docker + GitHub Actions + Codespaces

---

## 🌿 Estrategia de branches

| Branch | Propósito |
|--------|-----------|
| `dev` | Modelo base sin fine-tuning, desarrollo activo |
| `prod` | Modelo (fine-tuned cuando esté listo) en producción |
| `feature/*` | Nuevas funcionalidades, se mergean a `dev` |

**Flujo:** `feature/xxx` → PR → `dev` → PR → `prod`

---

## 🚀 Inicio rápido

### 1. Clonar y configurar
```bash
git clone https://github.com/dvilchess/lillama
cd tinyllama-finetune
cp .env.example .env.dev
```

### 2. Correr en DEV (Docker)
```bash
docker compose -f docker/docker-compose.dev.yml up --build
```
Abre http://localhost:7860

### 3. Correr en PROD
```bash
docker compose -f docker/docker-compose.prod.yml up --build
```

### 4. Descargar modelo manualmente
```bash
pip install huggingface_hub
python scripts/download_model.py
```

---

## 📓 Notebooks

| Notebook | Descripción |
|----------|-------------|
| `01_load_model_dev.ipynb` | Carga TinyLlama en Colab (sin GPU) |
| `02_finetune_colab.ipynb` | Fine-tuning con LoRA en Colab (GPU T4) |

---

## 🐳 Codespaces

Este repo incluye configuración para GitHub Codespaces.
Al abrir en Codespaces, el entorno dev se levanta automáticamente.

---

## 🔄 Cambiar a Dolphin (o cualquier otro modelo)

Solo cambia `MODEL_ID` en tu `.env.dev`:
```
MODEL_ID=cognitivecomputations/dolphin-2.9-llama3-8b
```

---

## 📁 Estructura

```
tinyllama-finetune/
├── app/
│   ├── config.py       # Config por ENV
│   ├── model.py        # Carga y generación
│   └── main.py         # Gradio UI
├── docker/
│   ├── Dockerfile.dev
│   ├── Dockerfile.prod
│   ├── docker-compose.dev.yml
│   └── docker-compose.prod.yml
├── notebooks/
│   ├── 01_load_model_dev.ipynb
│   └── 02_finetune_colab.ipynb
├── scripts/
│   └── download_model.py
├── tests/
├── .github/workflows/
│   ├── ci.yml
│   └── deploy.yml
├── .env.example
└── requirements.txt
```
