"""Text generation service (Transformers text-generation pipeline).

Lazy-loads GPT-2 / DistilGPT-2 on first request and caches it module-level.
"""
from flask import Blueprint, flash, render_template, request

import config

bp = Blueprint("textgen", __name__, url_prefix="/textgen")

_pipe = None  # module-level singleton cache


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("text-generation", model=config.TEXTGEN_MODEL)
    return _pipe


def _tidy(text: str) -> str:
    """Normalize GPT-2 output: trim, collapse blank-line runs and stray
    whitespace so the rendered continuation reads cleanly (no huge gaps)."""
    import re
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces/tabs
    text = re.sub(r"\n{2,}", "\n", text)        # collapse blank-line runs
    text = re.sub(r" *\n *", "\n", text)        # tidy spaces around newlines
    return text.strip()


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/textgen.html")

    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        flash("Please enter a prompt.", "error")
        return render_template("services/textgen.html")

    try:
        max_new = int(request.form.get("max_new_tokens", 60))
        temperature = float(request.form.get("temperature", 0.9))
        max_new = max(1, min(max_new, 250))            # clamp for safety
        temperature = max(0.1, min(temperature, 2.0))
    except ValueError:
        flash("max_new_tokens and temperature must be numbers.", "error")
        return render_template("services/textgen.html")

    try:
        out = _get_pipe()(
            prompt, max_new_tokens=max_new, temperature=temperature,
            do_sample=True, num_return_sequences=1, truncation=True,
            top_p=0.92, top_k=50,
            # Curb GPT-2's tendency to loop / repeat the same sentence.
            repetition_penalty=1.3, no_repeat_ngram_size=3,
        )
        full = out[0]["generated_text"]
        # Split the original prompt from the generated continuation.
        generated = full[len(prompt):] if full.startswith(prompt) else full
        generated = _tidy(generated)
    except Exception as exc:  # pragma: no cover
        flash(f"Generation failed: {exc}. Check internet / transformers "
              "version (model downloads on first use).", "error")
        return render_template("services/textgen.html")

    return render_template(
        "services/textgen.html",
        result={"prompt": prompt, "generated": generated,
                "model": config.TEXTGEN_MODEL,
                "max_new": max_new, "temperature": temperature},
    )
