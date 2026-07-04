"""DBSCAN clustering service.

Upload a CSV -> numeric columns are scaled -> DBSCAN is fit. Reports cluster
count, noise points, and silhouette score, plus a PCA-2D scatter plot.
"""
import os

from flask import Blueprint, flash, render_template, request

import config
from utils.files import load_csv, numeric_matrix, save_upload, scale_and_reduce
from utils.plotting import cluster_scatter

bp = Blueprint("dbscan", __name__, url_prefix="/dbscan")


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/dbscan.html")

    # --- parse params (with safe defaults) ---
    try:
        eps = float(request.form.get("eps", 0.5))
        min_samples = int(request.form.get("min_samples", 5))
        if eps <= 0 or min_samples < 1:
            raise ValueError
    except ValueError:
        flash("eps must be > 0 and min_samples a positive integer.", "error")
        return render_template("services/dbscan.html")

    # --- resolve input file (upload or bundled sample) ---
    try:
        if request.form.get("use_sample"):
            path = os.path.join(config.SAMPLE_DATA_DIR, "clustering_sample.csv")
        else:
            path = save_upload(request.files.get("csvfile"), config.ALLOWED_CSV)
        df = load_csv(path)
        num_df, cols = numeric_matrix(df)
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("services/dbscan.html")

    # --- fit DBSCAN ---
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score

        X, coords = scale_and_reduce(num_df)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

        n_clusters = len({lab for lab in labels if lab != -1})
        n_noise = int((labels == -1).sum())

        # Silhouette needs >=2 clusters and excludes noise-only edge cases.
        sil = None
        if n_clusters >= 2:
            mask = labels != -1
            if len(set(labels[mask])) >= 2:
                sil = round(float(silhouette_score(X[mask], labels[mask])), 3)

        # label -> count table
        import collections
        counts = collections.Counter(labels)
        table = [{"label": ("noise" if k == -1 else int(k)), "count": int(v)}
                 for k, v in sorted(counts.items())]

        plot = cluster_scatter(coords, labels,
                               f"DBSCAN (eps={eps}, min_samples={min_samples})",
                               noise_label=-1)
    except Exception as exc:  # pragma: no cover - guards bad data/model
        flash(f"Clustering failed: {exc}", "error")
        return render_template("services/dbscan.html")

    return render_template(
        "services/dbscan.html",
        result={
            "eps": eps, "min_samples": min_samples,
            "n_clusters": n_clusters, "n_noise": n_noise,
            "silhouette": sil, "columns": cols,
            "table": table, "plot": plot, "rows": len(labels),
        },
    )
