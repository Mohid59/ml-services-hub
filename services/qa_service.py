"""Voice Q&A service: audio question -> Whisper ASR -> extractive QA ->
gTTS speech of the answer.

Extractive QA needs a context passage; if the user leaves it blank we fall
back to a short built-in context (and say so). A text box for the question is
offered as a fallback alongside the audio input.
"""
from flask import Blueprint, flash, render_template, request

import config
from utils.audio import transcribe, tts_base64
from utils.files import save_upload

bp = Blueprint("qa", __name__, url_prefix="/qa")

# Load the QA model directly: the "question-answering" pipeline task was
# removed in transformers v5, so we run the model and extract the answer span
# manually. Works across v4 (grader) and v5 (here).
_tok = None
_model = None


def _get_model():
    global _tok, _model
    if _model is None:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(config.QA_MODEL)
        _model = AutoModelForQuestionAnswering.from_pretrained(config.QA_MODEL)
    return _tok, _model


def _answer(question: str, context: str):
    """Return (answer_text, confidence_0_1) for an extractive QA span."""
    import torch

    tok, model = _get_model()
    inp = tok(question, context, return_tensors="pt", truncation=True,
              max_length=512)
    with torch.no_grad():
        out = model(**inp)
    start = int(out.start_logits.argmax())
    end = int(out.end_logits.argmax())
    if end < start:
        end = start
    ids = inp["input_ids"][0][start:end + 1]
    answer = tok.decode(ids, skip_special_tokens=True).strip()
    # Confidence: product of softmax-normalized start/end probabilities.
    s_prob = float(torch.softmax(out.start_logits, dim=1)[0, start])
    e_prob = float(torch.softmax(out.end_logits, dim=1)[0, end])
    return (answer or "(no answer found)"), s_prob * e_prob


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/qa.html",
                               default_context=config.DEFAULT_QA_CONTEXT)

    # --- context (user-provided or built-in fallback) ---
    context = (request.form.get("context") or "").strip()
    used_default_context = False
    if not context:
        context = config.DEFAULT_QA_CONTEXT
        used_default_context = True

    # --- question from audio (preferred) or text fallback ---
    source = "audio"
    question = ""
    try:
        audio = request.files.get("audiofile")
        if audio and audio.filename:
            path = save_upload(audio, config.ALLOWED_AUDIO)
            question = transcribe(path)
        else:
            question = (request.form.get("question") or "").strip()
            source = "text"
        if not question:
            flash("Provide an audio question or type one.", "error")
            return render_template("services/qa.html",
                                   default_context=context)
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("services/qa.html", default_context=context)
    except Exception as exc:  # pragma: no cover
        flash(f"Transcription failed: {exc}. Is ffmpeg installed?", "error")
        return render_template("services/qa.html", default_context=context)

    # --- answer + speech ---
    try:
        answer, conf = _answer(question, context)
        score = round(conf * 100, 1)
        audio_b64 = tts_base64(answer)         # gTTS -> base64 MP3
    except Exception as exc:  # pragma: no cover
        flash(f"Q&A failed: {exc}.", "error")
        return render_template("services/qa.html", default_context=context)

    return render_template(
        "services/qa.html", default_context=context,
        result={"question": question, "answer": answer, "score": score,
                "source": source, "audio": audio_b64,
                "used_default_context": used_default_context},
    )
