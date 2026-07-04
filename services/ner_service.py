"""Named Entity Recognition service (dslim/bert-base-NER).

Uses aggregation_strategy='simple' so sub-word tokens are merged into whole
entities. Builds a list of text spans (entity vs plain) for inline
highlighting in the template, plus a table of entities.
"""
from flask import Blueprint, flash, render_template, request

import config

bp = Blueprint("ner", __name__, url_prefix="/ner")

_pipe = None  # module-level singleton cache

_DEFAULT_TEXT = ("Barack Obama was born in Hawaii and worked at Microsoft "
                 "before visiting Paris with the United Nations.")


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("ner", model=config.NER_MODEL,
                         aggregation_strategy="simple")
    return _pipe


def _build_spans(text, entities):
    """Split text into ordered spans, tagging entity ranges with their type."""
    spans, cursor = [], 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        start, end = ent["start"], ent["end"]
        if start > cursor:
            spans.append({"text": text[cursor:start], "type": None})
        spans.append({"text": text[start:end], "type": ent["entity_group"]})
        cursor = end
    if cursor < len(text):
        spans.append({"text": text[cursor:], "type": None})
    return spans


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/ner.html", default_text=_DEFAULT_TEXT)

    text = (request.form.get("text") or "").strip()
    if not text:
        flash("Please enter some text.", "error")
        return render_template("services/ner.html", default_text=_DEFAULT_TEXT)

    try:
        raw = _get_pipe()(text)
        entities = [{
            "start": int(e["start"]), "end": int(e["end"]),
            "entity_group": e["entity_group"],
            "word": e["word"], "score": round(float(e["score"]), 3),
        } for e in raw]
        spans = _build_spans(text, entities)
    except Exception as exc:  # pragma: no cover
        flash(f"NER failed: {exc}. Check internet / transformers version.",
              "error")
        return render_template("services/ner.html", default_text=_DEFAULT_TEXT)

    return render_template(
        "services/ner.html", default_text=text,
        result={"spans": spans, "entities": entities,
                "n": len(entities), "model": config.NER_MODEL},
    )
