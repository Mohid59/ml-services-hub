"""Pre-download every transformer model into the HF cache.

Run at Docker build time so the image ships with all weights baked in:
first request per service is instant even after container restarts
(Spaces free tier wipes runtime disk on every rebuild/wake).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from huggingface_hub import snapshot_download  # noqa: E402

import config  # noqa: E402

MODELS = [
    config.WHISPER_MODEL,
    config.SENTIMENT_MODEL,
    config.QA_MODEL,
    config.TEXTGEN_MODEL,
    config.TRANSLATION_MODEL,
    config.NER_MODEL,
]

for repo in MODELS:
    print(f"[preload] {repo}", flush=True)
    snapshot_download(repo)

print("[preload] all models cached", flush=True)
