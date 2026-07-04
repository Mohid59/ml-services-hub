"""Apriori association-rule mining service.

Accepts two CSV shapes and auto-detects which:
  (a) transactional - each row is a basket; items spread across columns
      and/or comma-separated inside cells.
  (b) one-hot encoded - all columns numeric/boolean with 0/1 values.
Runs mlxtend apriori + association_rules, sorted by lift desc.
"""
import os

import pandas as pd
from flask import Blueprint, flash, render_template, request

import config
from utils.files import load_csv, save_upload

bp = Blueprint("apriori", __name__, url_prefix="/apriori")


def _looks_one_hot(df: pd.DataFrame) -> bool:
    """True if every column is numeric with values only in {0, 1}."""
    num = df.select_dtypes(include="number")
    if num.shape[1] != df.shape[1] or df.shape[1] == 0:
        return False
    vals = pd.unique(num.values.ravel())
    return set(pd.Series(vals).dropna().unique()).issubset({0, 1})


def _rows_to_transactions(df: pd.DataFrame):
    """Convert a transactional DataFrame into a list of item lists.

    Collects every non-empty cell per row, splitting comma-separated cells.
    """
    transactions = []
    for _, row in df.iterrows():
        items = []
        for cell in row:
            if pd.isna(cell):
                continue
            text = str(cell).strip()
            if not text:
                continue
            items.extend(p.strip() for p in text.split(",") if p.strip())
        if items:
            transactions.append(sorted(set(items)))
    return transactions


def _build_basket(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean one-hot basket DataFrame from any accepted shape."""
    if _looks_one_hot(df):
        return df.astype(bool)
    from mlxtend.preprocessing import TransactionEncoder
    transactions = _rows_to_transactions(df)
    if not transactions:
        raise ValueError("No items found in the transactional CSV.")
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


def _fmt_set(frozen) -> str:
    """Pretty-print a frozenset of items as 'a, b'."""
    return ", ".join(sorted(str(x) for x in frozen))


@bp.route("/", methods=["GET", "POST"])
def page():
    if request.method == "GET":
        return render_template("services/apriori.html")

    # --- params ---
    try:
        min_support = float(request.form.get("min_support", 0.05))
        min_conf = float(request.form.get("min_confidence", 0.5))
        if not (0 < min_support <= 1) or not (0 < min_conf <= 1):
            raise ValueError
    except ValueError:
        flash("min_support and min_confidence must be in (0, 1].", "error")
        return render_template("services/apriori.html")

    # --- input ---
    try:
        if request.form.get("use_sample"):
            path = os.path.join(config.SAMPLE_DATA_DIR,
                                "transactions_sample.csv")
        else:
            path = save_upload(request.files.get("csvfile"), config.ALLOWED_CSV)
        df = load_csv(path)
        basket = _build_basket(df)
        shape = "one-hot" if _looks_one_hot(df) else "transactional"
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("services/apriori.html")

    # --- mine ---
    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        freq = apriori(basket, min_support=min_support, use_colnames=True)
        if freq.empty:
            flash(f"No frequent itemsets at min_support={min_support}. "
                  "Try lowering it.", "info")
            return render_template("services/apriori.html",
                                   meta={"shape": shape, "n_tx": len(basket)})

        rules = association_rules(freq, metric="confidence",
                                 min_threshold=min_conf)
        if rules.empty:
            flash(f"Found itemsets but no rules at min_confidence={min_conf}. "
                  "Try lowering it.", "info")
            return render_template("services/apriori.html",
                                   meta={"shape": shape, "n_tx": len(basket)})

        rules = rules.sort_values("lift", ascending=False)
        table = [{
            "antecedents": _fmt_set(r.antecedents),
            "consequents": _fmt_set(r.consequents),
            "support": round(float(r.support), 3),
            "confidence": round(float(r.confidence), 3),
            "lift": round(float(r.lift), 3),
        } for r in rules.itertuples(index=False)]
    except Exception as exc:  # pragma: no cover
        flash(f"Rule mining failed: {exc}", "error")
        return render_template("services/apriori.html")

    return render_template(
        "services/apriori.html",
        result={
            "min_support": min_support, "min_confidence": min_conf,
            "shape": shape, "n_tx": len(basket),
            "n_items": basket.shape[1], "n_rules": len(table),
            "table": table,
        },
    )
