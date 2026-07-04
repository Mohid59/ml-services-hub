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

EXPOSE 5000
ENV FLASK_ENV=production PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/healthz')"

CMD ["python", "app.py"]
