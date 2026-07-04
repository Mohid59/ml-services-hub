# ML Services Hub — CPU-only image
FROM python:3.11-slim

# ffmpeg: required by Whisper for mp3/m4a decoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights download lazily into the HF cache on first request;
# mount a volume here to persist them across container restarts.
VOLUME ["/root/.cache/huggingface"]

# 7860 = Hugging Face Spaces convention (works anywhere)
EXPOSE 7860
ENV FLASK_ENV=production PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/healthz')"

# Single worker: models are cached as module singletons; threads share them.
# Long timeout covers first-request model downloads.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", \
     "--threads", "4", "--timeout", "600", "app:create_app()"]
