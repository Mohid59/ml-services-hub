"""Safe upload handling + CSV loading/validation shared across services."""
import os

import pandas as pd
from werkzeug.utils import secure_filename

import config


def ext_of(filename: str) -> str:
    """Lowercase file extension including the dot, or '' if none."""
    return os.path.splitext(filename or "")[1].lower()


def save_upload(file_storage, allowed_exts) -> str:
    """Validate and save an uploaded file to the uploads dir.

    Returns the saved absolute path. Raises ValueError with a friendly
    message on any validation failure (empty / wrong type).
    """
    if file_storage is None or file_storage.filename == "":
        raise ValueError("No file was selected.")

    ext = ext_of(file_storage.filename)
    if ext not in allowed_exts:
        allowed = ", ".join(sorted(allowed_exts))
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {allowed}")

    safe_name = secure_filename(file_storage.filename)
    dest = os.path.join(config.UPLOADS_DIR, safe_name)
    file_storage.save(dest)
    return dest


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame with a friendly error on failure."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - depends on user file
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if df.empty:
        raise ValueError("The CSV file is empty.")
    return df


def numeric_matrix(df: pd.DataFrame):
    """Select numeric columns, drop NaN rows. Returns (X_df, used_columns).

    Raises ValueError if fewer than 2 usable numeric columns remain.
    """
    num = df.select_dtypes(include="number").dropna()
    if num.shape[1] < 2:
        raise ValueError(
            "Need at least 2 numeric columns for clustering. "
            "This CSV has " + str(num.shape[1]) + "."
        )
    if num.shape[0] < 3:
        raise ValueError("Need at least 3 rows after dropping NaNs.")
    return num, list(num.columns)


def scale_and_reduce(num_df):
    """Standard-scale a numeric DataFrame and compute a 2D PCA projection.

    Returns (X_scaled, coords_2d) where coords_2d is an (n, 2) array used
    for plotting. PCA is only for visualization; clustering runs on the
    full scaled feature space.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(num_df.values)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    return X, coords
