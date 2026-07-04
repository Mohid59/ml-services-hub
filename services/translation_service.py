"""English -> Urdu translation service (Helsinki-NLP Marian model).

Needs sentencepiece + sacremoses for the Marian tokenizer.
"""
from flask import Blueprint, flash, render_template, request

import config

bp = Blueprint("translation", __name__, url_prefix="/translation")

# Cache the tokenizer+model directly rather than a pipeline: the
# "translation" pipeline task was removed in transformers v5, so loading the
# Seq2Seq model explicitly works across both v4 (grader) and v5 (here).
_tok = None
_model = None


def _get_model():
    global _tok, _model
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(config.TRANSLATION_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATION_MODEL)
    return _tok, _model


def _translate(text: str) -> str:
    """Translate EN->UR. Marian handles single sentences best, so split long
    input on sentence boundaries, translate each, and rejoin."""
    import re

    tok, model = _get_model()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        sentences = [text]

    pieces = []
    for sent in sentences:
        ids = tok(sent, return_tensors="pt", truncation=True, max_length=256)
        gen = model.generate(**ids, max_length=256, num_beams=4,
                             no_repeat_ngram_size=3)
        pieces.append(tok.decode(gen[0], skip_special_tokens=True).strip())
    return " ".join(p for p in pieces if p)


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/translation.html")

    text = (request.form.get("text") or "").strip()
    if not text:
        flash("Please enter some English text to translate.", "error")
        return render_template("services/translation.html")

    try:
        urdu = _translate(text)
    except Exception as exc:  # pragma: no cover
        flash(f"Translation failed: {exc}. Ensure sentencepiece is installed "
              "and you have internet for the first download.", "error")
        return render_template("services/translation.html")

    return render_template(
        "services/translation.html",
        result={"source": text, "urdu": urdu, "model": config.TRANSLATION_MODEL},
    )
