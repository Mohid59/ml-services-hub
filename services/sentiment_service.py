"""Voice sentiment service: audio -> Whisper ASR -> DistilBERT sentiment.

Primary input is an audio clip. A text box is also offered as a fallback so
the service is demoable without a microphone / ffmpeg.
"""
from flask import Blueprint, flash, render_template, request

import config
from utils.audio import transcribe
from utils.files import save_upload

bp = Blueprint("sentiment", __name__, url_prefix="/sentiment")

_pipe = None  # module-level singleton cache


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL)
    return _pipe


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/sentiment.html")

    source = "audio"
    text = ""
    try:
        audio = request.files.get("audiofile")
        if audio and audio.filename:
            path = save_upload(audio, config.ALLOWED_AUDIO)
            text = transcribe(path)            # Whisper ASR
        else:
            text = (request.form.get("text") or "").strip()
            source = "text"
        if not text:
            flash("Provide an audio clip or some text.", "error")
            return render_template("services/sentiment.html")
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("services/sentiment.html")
    except Exception as exc:  # pragma: no cover - ASR/runtime issues
        flash(f"Transcription failed: {exc}. Is ffmpeg installed?", "error")
        return render_template("services/sentiment.html")

    try:
        res = _get_pipe()(text)[0]
    except Exception as exc:  # pragma: no cover
        flash(f"Sentiment analysis failed: {exc}.", "error")
        return render_template("services/sentiment.html")

    return render_template(
        "services/sentiment.html",
        result={"transcript": text, "source": source,
                "label": res["label"],
                "score": round(float(res["score"]) * 100, 1)},
    )
