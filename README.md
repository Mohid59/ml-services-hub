---
title: ML Services Hub
emoji: 🧪
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 9 ML services behind one Flask UI
---

# ML Services Hub

**🔴 Live demo: [mohid59-ml-services-hub.hf.space](https://mohid59-ml-services-hub.hf.space)** · hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Mohid59/ml-services-hub)

[![Live Demo](https://img.shields.io/badge/🤗%20Spaces-Live_Demo-yellow)](https://huggingface.co/spaces/Mohid59/ml-services-hub)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

A production-style Flask application exposing **nine machine-learning services** behind one
editorial-styled web UI — classic ML (clustering, association rules), computer vision
(a CNN I trained on UTKFace), and transformer-based NLP & speech (Whisper ASR, extractive QA
with spoken answers, EN→Urdu translation, NER, GPT-2 generation).

**Highlights**

- 🧠 **9 live ML services**, every heavy model lazy-loaded and cached (fast boot, CPU-only)
- 🎓 **Custom-trained CNN** for gender classification (UTKFace, balanced subset, augmentation)
  with a graceful MobileNetV2 fallback when no trained weights exist
- 🔊 **Voice in / voice out** pipelines: Whisper → DistilBERT sentiment; Whisper → SQuAD QA → gTTS
- ✅ **36-test pytest suite** (fast tier for CI, `slow` tier for model-backed paths), ruff-clean
- 🐳 **Dockerfile** with healthcheck + `/healthz` liveness endpoint + structured request logging
- 🔁 **GitHub Actions CI**: lint + fast tests on every push
- 🛡️ Defensive input handling everywhere: wrong file types, corrupt CSVs, empty text, oversized
  uploads → friendly flash messages, never a stack trace
- 🔄 **Version-robust transformers integration**: translation & QA use `AutoModel*` classes
  directly, working on both transformers 4.x and 5.x (the pipeline task names were removed in v5)

---

## Services

| # | Service | Pipeline | Model |
|---|---------|----------|-------|
| 1 | DBSCAN Clustering | CSV → metrics + PCA scatter | scikit-learn `DBSCAN` |
| 2 | K-Means Clustering | CSV → metrics + scatter + elbow | scikit-learn `KMeans` |
| 3 | Apriori Rules | basket/one-hot CSV (auto-detected) → rules by lift | mlxtend |
| 4 | Gender Classifier | image → label + confidence | **custom Keras CNN** (UTKFace) |
| 5 | Voice Sentiment | audio → transcript → POSITIVE/NEGATIVE | Whisper-base + DistilBERT SST-2 |
| 6 | Voice Q&A | audio question → answer text **+ spoken audio** | Whisper + DistilBERT SQuAD + gTTS |
| 7 | Text Generation | prompt → continuation (anti-repetition tuned) | DistilGPT-2 |
| 8 | EN → Urdu | text → RTL Urdu (sentence-split for quality) | Helsinki-NLP opus-mt-en-ur |
| 9 | NER | text → inline color-coded entities + table | dslim/bert-base-NER |

## Quickstart

```bash
git clone <this-repo> && cd <this-repo>
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py                          # → http://localhost:5000
```

First request per ML service downloads its weights from the Hugging Face Hub
(cached afterwards). The app itself boots in ~1 second because every model is lazy-loaded.

### Docker

```bash
docker build -t ml-services-hub .
docker run -p 5000:5000 -v hf-cache:/root/.cache/huggingface ml-services-hub
```

The named volume persists downloaded model weights across restarts.
`/healthz` serves as the container healthcheck.

### ffmpeg (audio services)

Whisper decodes `.mp3`/`.m4a` via ffmpeg (bundled in the Docker image):

- Windows: `winget install Gyan.FFmpeg` · macOS: `brew install ffmpeg` · Linux: `apt install ffmpeg`

## Training the gender CNN

The repo ships two scripts that take you from nothing to a trained model:

```bash
pip install -r requirements-dev.txt

# 1. Stream a balanced UTKFace subset from the HF Hub (no 1 GB download)
python services/prepare_gender_data.py --per-class 2400

# 2. Train the CNN (3×Conv/MaxPool → Dense → sigmoid, augmentation, 64×64 RGB)
python services/train_gender_cnn.py --data data --epochs 12
```

This writes `models/gender_cnn.h5`, which the web service auto-loads on the next request.
Without it, the service falls back to an untrained MobileNetV2 head and **clearly labels
the prediction as a placeholder** — the demo never breaks, and never lies.

### Model card (gender CNN)

| | |
|---|---|
| Data | UTKFace via `nu-delta/utkface` (HF Hub), shuffled stream, 2,400/class |
| Input | 64×64 RGB, rescaled 1/255, flip/rotate/zoom augmentation |
| Architecture | Conv32-Conv64-Conv128 (each + MaxPool) → Dropout 0.4 → Dense 128 → sigmoid |
| Split | 90% train / 10% val, label order pinned `["female", "male"]` |
| Result | **85–86% val accuracy** after 12 epochs (~4 min CPU); 83.8% re-measured end-to-end through the web service's own preprocessing on 240 held-out faces |
| Intended use | Course demo of an end-to-end CV pipeline. Binary labels reflect the dataset's annotations; not suitable for production identity decisions. |

## Tests & quality

```bash
pytest -m "not slow"   # fast tier: routing, validation, classic ML (CI runs this)
pytest                 # full tier: transformer/CNN happy paths (downloads models)
ruff check .           # lint (config in pyproject.toml)
```

CI (`.github/workflows/ci.yml`) runs lint + the fast tier on every push/PR to `main`.

## Architecture

```
app.py                  application factory · blueprint registry · /healthz · request logging
config.py               single source of truth: paths, limits, model ids
services/               one Flask blueprint per ML service
  ├── *_service.py        lazy singleton model loading + defensive request handling
  ├── prepare_gender_data.py   UTKFace streaming downloader
  └── train_gender_cnn.py      standalone Keras training script
utils/
  ├── plotting.py         headless matplotlib → base64 PNG
  ├── files.py            secure uploads · CSV validation · scale + PCA helpers
  └── audio.py            Whisper ASR + gTTS synthesis (base64 MP3)
templates/ + static/    server-rendered Jinja2 · hand-built editorial design system
tests/                  pytest suite (fast/slow tiers)
```

**Design decisions worth noting**

- *Lazy singletons per service*: each model loads on first request and is cached at module
  level — boot stays instant, memory is only paid for services actually used.
- *Graceful degradation as a feature*: missing trained weights, missing ffmpeg, bad uploads —
  every failure mode has a designed UX, not an exception page.
- *No frontend build step*: server-rendered Jinja2 + one hand-written CSS system
  (Fraunces/JetBrains Mono, print-catalogue aesthetic). Zero JS frameworks.
- *Reproducibility*: `random_state=42` across sklearn; pinned requirements.

## License

MIT
