"""Central configuration for ML Services Hub.

All paths are derived from this file's location so the app has no
hardcoded absolute paths and runs from any clone location.
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")

# Ensure runtime dirs exist (created at import time, safe to repeat)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Upload limits
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB cap

ALLOWED_CSV = {".csv"}
ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

# Model ids / constants (small, CPU-friendly variants)
WHISPER_MODEL = "openai/whisper-base"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
QA_MODEL = "distilbert-base-cased-distilled-squad"
TEXTGEN_MODEL = "distilgpt2"          # swap to "gpt2" for higher quality
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-ur"
NER_MODEL = "dslim/bert-base-NER"

GENDER_CNN_PATH = os.path.join(MODELS_DIR, "gender_cnn.h5")
GENDER_IMG_SIZE = (64, 64)

# Default built-in QA context when user supplies none
DEFAULT_QA_CONTEXT = (
    "Machine learning is a branch of artificial intelligence focused on "
    "building systems that learn from data. Flask is a lightweight Python "
    "web framework. The ML Services Hub demonstrates nine machine learning "
    "services including clustering, association rule mining, image "
    "classification, and several transformer-based NLP tasks."
)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-ml-services-hub-key")
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH
    UPLOAD_FOLDER = UPLOADS_DIR
