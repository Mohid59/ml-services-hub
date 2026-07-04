"""K-Means clustering service.

Same upload/preprocessing pipeline as DBSCAN. Reports inertia + silhouette,
draws a PCA-2D scatter with centroids, and an elbow plot (k=2..10).
"""
import os

from flask import Blueprint, flash, render_template, request

import config
from utils.files import load_csv, numeric_matrix, save_upload, scale_and_reduce
from utils.plotting import cluster_scatter, elbow_plot

bp = Blueprint("kmeans", __name__, url_prefix="/kmeans")


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/kmeans.html")

    # --- parse k ---
    try:
        k = int(request.form.get("k", 3))
        if k < 2:
            raise ValueError
    except ValueError:
        flash("k must be an integer >= 2.", "error")
        return render_template("services/kmeans.html")

    # --- resolve input ---
    try:
        if request.form.get("use_sample"):
            path = os.path.join(config.SAMPLE_DATA_DIR, "clustering_sample.csv")
        else:
            path = save_upload(request.files.get("csvfile"), config.ALLOWED_CSV)
        df = load_csv(path)
        num_df, cols = numeric_matrix(df)
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("services/kmeans.html")

    # --- fit ---
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score

        X, coords = scale_and_reduce(num_df)
        if k > len(X):
            flash(f"k ({k}) cannot exceed number of rows ({len(X)}).", "error")
            return render_template("services/kmeans.html")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertia = round(float(km.inertia_), 2)
        sil = round(float(silhouette_score(X, labels)), 3) if k >= 2 else None

        # Project centroids into the same PCA space for plotting.
        pca = PCA(n_components=2, random_state=42).fit(X)
        cent_2d = pca.transform(km.cluster_centers_)

        import collections
        counts = collections.Counter(labels)
        table = [{"label": int(kk), "count": int(v)}
                 for kk, v in sorted(counts.items())]

        scatter = cluster_scatter(coords, labels, f"K-Means (k={k})",
                                  centroids=cent_2d)

        # Elbow: inertia across k=2..min(10, n-1)
        ks = range(2, min(10, len(X) - 1) + 1)
        inertias = [KMeans(n_clusters=kk, random_state=42, n_init=10)
                    .fit(X).inertia_ for kk in ks]
        elbow = elbow_plot(list(ks), inertias)
    except Exception as exc:  # pragma: no cover
        flash(f"Clustering failed: {exc}", "error")
        return render_template("services/kmeans.html")

    return render_template(
        "services/kmeans.html",
        result={
            "k": k, "inertia": inertia, "silhouette": sil,
            "columns": cols, "table": table, "rows": len(labels),
            "scatter": scatter, "elbow": elbow,
        },
    )
