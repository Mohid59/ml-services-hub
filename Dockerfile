# ML Services Hub — CPU-only image with all model weights baked in
FROM python:3.11-slim

# ffmpeg: required by Whisper for mp3/m4a decoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake all transformer weights into the image (own layer: only re-runs when
# config.py model ids change, not on every code edit). This makes first
# requests instant even after Space rebuilds, which wipe the runtime disk.
COPY config.py scripts/preload_models.py ./scripts_preload/
RUN python scripts_preload/preload_models.py && rm -rf scripts_preload

COPY . .

# 7860 = Hugging Face Spaces convention (works anywhere)
EXPOSE 7860
ENV FLASK_ENV=production PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/healthz')"

# Single worker: models are cached as module singletons; threads share them.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", \
     "--threads", "4", "--timeout", "600", "app:create_app()"]
