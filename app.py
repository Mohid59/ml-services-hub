"""ML Services Hub - Flask app factory.

Run with:  python app.py
Boots fast: all heavy ML models are lazy-loaded on first request, not here.
"""
import logging
import time

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from config import Config

# Service metadata drives the dashboard cards and nav. Keeping it in one
# place means adding a service is a single edit here + one blueprint.
SERVICES = [
    # slug, display name, one-line description, emoji icon, group
    ("dbscan", "DBSCAN Clustering", "Density-based clustering of CSV data.",
     "\U0001F310", "Clustering & Rules"),
    ("kmeans", "K-Means Clustering", "Partition CSV data into k clusters.",
     "\U0001F4CA", "Clustering & Rules"),
    ("apriori", "Apriori Rules", "Mine association rules from baskets.",
     "\U0001F6D2", "Clustering & Rules"),
    ("gender", "Gender Classifier", "CNN predicts gender from a face image.",
     "\U0001F9D1", "Computer Vision"),
    ("sentiment", "Voice Sentiment", "Speak → transcribe → sentiment.",
     "\U0001F3A4", "Speech"),
    ("qa", "Voice Q&A", "Ask by voice, hear the answer.",
     "\U0001F5E3️", "Speech"),
    ("textgen", "Text Generation", "Continue a prompt with GPT-2.",
     "✍️", "NLP / Transformers"),
    ("translation", "EN → Urdu", "Translate English to Urdu.",
     "\U0001F310", "NLP / Transformers"),
    ("ner", "Named Entities", "Highlight people, places, orgs in text.",
     "\U0001F3F7️", "NLP / Transformers"),
]

# Display order of the groups on the dashboard.
GROUP_ORDER = ["Clustering & Rules", "Computer Vision",
               "NLP / Transformers", "Speech"]


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("ml_hub")

    _register_blueprints(app)

    @app.before_request
    def _start_timer():
        request._t0 = time.perf_counter()

    @app.after_request
    def _log_request(response):
        # Structured one-line access log with latency - useful in Docker/CI.
        dt = (time.perf_counter() - getattr(request, "_t0", time.perf_counter()))
        log.info("%s %s -> %s (%.0f ms)", request.method, request.path,
                 response.status_code, dt * 1000)
        return response

    @app.route("/healthz")
    def healthz():
        """Liveness probe for containers/orchestrators.

        Also reports whether the trained gender model shipped - a missing
        file silently degrades that service to an untrained fallback, so
        surface it where a deploy check will see it.
        """
        import os

        import config as cfg
        return jsonify(status="ok", services=len(SERVICES),
                       gender_model="trained" if os.path.exists(cfg.GENDER_CNN_PATH)
                       else "fallback")

    @app.route("/")
    def index():
        # Group services for the grouped card layout.
        grouped = {g: [] for g in GROUP_ORDER}
        for slug, name, desc, icon, group in SERVICES:
            grouped[group].append(
                {"slug": slug, "name": name, "desc": desc, "icon": icon}
            )
        return render_template("index.html", grouped=grouped,
                               group_order=GROUP_ORDER)

    @app.errorhandler(413)
    def too_large(_e):
        flash("Upload too large — the limit is 16 MB.", "error")
        return redirect(url_for("index"))

    @app.errorhandler(404)
    def not_found(_e):
        flash("Page not found.", "error")
        return redirect(url_for("index"))

    return app


def _register_blueprints(app):
    """Import and register every service blueprint.

    Imports are local so a failure in one optional-dependency service does
    not stop the whole app from booting.
    """
    from services.apriori_service import bp as apriori_bp
    from services.dbscan_service import bp as dbscan_bp
    from services.gender_cnn_service import bp as gender_bp
    from services.kmeans_service import bp as kmeans_bp
    from services.ner_service import bp as ner_bp
    from services.qa_service import bp as qa_bp
    from services.sentiment_service import bp as sentiment_bp
    from services.textgen_service import bp as textgen_bp
    from services.translation_service import bp as translation_bp

    for bp in (dbscan_bp, kmeans_bp, apriori_bp, gender_bp, sentiment_bp,
               qa_bp, textgen_bp, translation_bp, ner_bp):
        app.register_blueprint(bp)


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
