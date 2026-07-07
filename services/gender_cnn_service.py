"""Gender image classification service (Keras CNN).

Inference strategy:
  * If models/gender_cnn.h5 exists (produced by train_gender_cnn.py), load
    and use that trained CNN.
  * Otherwise fall back to an untrained MobileNetV2 base + 2-class head so the
    page is still demoable, and print a console warning. The fallback's
    predictions are NOT meaningful until a real model is trained - this is
    surfaced in the UI.
"""
import os

import numpy as np
from flask import Blueprint, flash, render_template, request, send_from_directory

import config
from utils.files import save_upload

bp = Blueprint("gender", __name__, url_prefix="/gender")

_model = None
_is_fallback = False  # True when using the untrained MobileNetV2 fallback
CLASSES = ["female", "male"]  # index 0 / 1 (alphabetical, matches Keras flow)


def _build_fallback():
    """MobileNetV2 base + a fresh 2-class head (untrained)."""
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=(*config.GENDER_IMG_SIZE, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ])
    return model


def _get_model():
    """Lazily load the trained CNN, else build the fallback. Cached."""
    global _model, _is_fallback
    if _model is None:
        from tensorflow.keras.models import load_model
        if os.path.exists(config.GENDER_CNN_PATH):
            _model = load_model(config.GENDER_CNN_PATH)
            _is_fallback = False
            print(f"[gender] Loaded trained CNN: {config.GENDER_CNN_PATH}")
        else:
            print("[gender] WARNING: models/gender_cnn.h5 not found - using "
                  "UNTRAINED MobileNetV2 fallback. Train with "
                  "services/train_gender_cnn.py for real predictions.")
            _model = _build_fallback()
            _is_fallback = True
    return _model


def _input_size(model):
    """Read (W, H) the model expects; falls back to the config default."""
    shape = getattr(model, "input_shape", None)
    if shape and len(shape) == 4 and shape[1] and shape[2]:
        return (shape[2], shape[1])  # PIL wants (width, height)
    return config.GENDER_IMG_SIZE


def _preprocess(path, model=None):
    """Load image -> (1, H, W, 3) float array scaled to [0, 1].

    Input size is derived from the loaded model, so retraining at a
    different --img-size needs no code change here.
    """
    from PIL import Image
    size = _input_size(model) if model is not None else config.GENDER_IMG_SIZE
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


@bp.route("/uploads/<path:name>")
def uploaded(name):
    """Serve a previously uploaded image for the result preview."""
    return send_from_directory(config.UPLOADS_DIR, name)


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/gender.html")

    try:
        path = save_upload(request.files.get("imagefile"), config.ALLOWED_IMAGE)
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("services/gender.html")

    try:
        model = _get_model()
        x = _preprocess(path, model)
        preds = model.predict(x, verbose=0)[0]
        # Handle both 2-unit softmax and single-unit sigmoid heads.
        if preds.shape[-1] == 1:
            prob_male = float(preds[0])
            probs = [1 - prob_male, prob_male]
        else:
            probs = [float(p) for p in preds]
        idx = int(np.argmax(probs))
        label = CLASSES[idx]
        confidence = round(probs[idx] * 100, 1)
    except Exception as exc:  # pragma: no cover
        flash(f"Prediction failed: {exc}.", "error")
        return render_template("services/gender.html")

    # Web-path for previewing the uploaded image.
    rel = os.path.basename(path)
    return render_template(
        "services/gender.html",
        result={"label": label, "confidence": confidence,
                "fallback": _is_fallback, "filename": rel},
    )
