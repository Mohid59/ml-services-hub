"""Matplotlib helpers: render figures to base64 PNG for inline <img> embeds.

Never call plt.show() - the app is headless. We force the non-interactive
'Agg' backend so figures render without a display server.
"""
import base64
import io

import matplotlib

matplotlib.use("Agg")  # headless backend - must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402


def fig_to_base64(fig) -> str:
    """Serialize a Matplotlib figure to a base64 PNG data URI body.

    Returns the base64 string (no data: prefix). Caller embeds it as:
        <img src="data:image/png;base64,{{ value }}">
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)  # free memory - critical in a long-running server
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def cluster_scatter(coords, labels, title, centroids=None, noise_label=None):
    """PCA-2D scatter colored by cluster label -> base64 PNG.

    coords: (n, 2) array. labels: cluster id per point. centroids: optional
    (k, 2) array to mark with X. noise_label: if set (DBSCAN's -1), those
    points are drawn black.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(uniq):
        mask = np.asarray(labels) == lab
        if noise_label is not None and lab == noise_label:
            ax.scatter(coords[mask, 0], coords[mask, 1], c="black",
                       s=28, alpha=0.6, label="noise")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=cmap(i % 10), s=40, alpha=0.8,
                       label=f"cluster {lab}")
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=180, edgecolors="white", linewidths=1.5, label="centroids")
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8, loc="best")
    return fig_to_base64(fig)


def elbow_plot(ks, inertias):
    """Line plot of inertia vs k for the K-Means elbow method -> base64 PNG."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ks), inertias, "o-", color="#6C4DF6")
    ax.set_title("Elbow Method (inertia vs k)")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)
