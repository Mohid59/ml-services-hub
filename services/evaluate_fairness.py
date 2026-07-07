"""Fairness audit for the gender classifier.

Streams UTKFace with the SAME shuffle seed as prepare_gender_data.py, skips
exactly the images that went into train/val, and evaluates the saved model on
the next N unseen faces per gender. Reports accuracy broken down by the
dataset's gender and ethnicity annotations, so demographic performance gaps
are measured instead of assumed.

Run (after training):
    python services/evaluate_fairness.py --skip-per-class 2400 --eval-per-class 300

Prints a markdown table ready to paste into the README model card.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Audit gender model fairness")
    parser.add_argument("--skip-per-class", type=int, default=2400,
                        help="images per class consumed by prepare_gender_data")
    parser.add_argument("--eval-per-class", type=int, default=300)
    parser.add_argument("--model", default=config.GENDER_CNN_PATH)
    args = parser.parse_args()

    from datasets import load_dataset
    from tensorflow.keras.models import load_model

    model = load_model(args.model)
    shape = model.input_shape  # (None, H, W, 3)
    size = (shape[2], shape[1])
    print(f"Model: {args.model}  input {size}")

    # Same seed + buffer as the prep script -> identical stream order, so the
    # first `skip_per_class` images per gender are exactly the training pool.
    ds = load_dataset("nu-delta/utkface", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=4000)

    seen = {"female": 0, "male": 0}
    used = {"female": 0, "male": 0}
    # stats[(gender, ethnicity)] = [correct, total]
    stats = defaultdict(lambda: [0, 0])

    for ex in ds:
        gender = str(ex["gender"]).strip().lower()
        if gender not in seen:
            continue
        seen[gender] += 1
        if seen[gender] <= args.skip_per_class:
            continue                     # was available to training - skip
        if used[gender] >= args.eval_per_class:
            if all(v >= args.eval_per_class for v in used.values()):
                break
            continue

        img = ex["image"].convert("RGB").resize(size)
        x = np.expand_dims(np.asarray(img, dtype="float32") / 255.0, 0)
        prob_male = float(model.predict(x, verbose=0)[0][0])
        pred = "male" if prob_male >= 0.5 else "female"

        eth = str(ex["ethnicity"]).strip() or "Unknown"
        correct = int(pred == gender)
        stats[(gender, eth)][0] += correct
        stats[(gender, eth)][1] += 1
        stats[(gender, "ALL")][0] += correct
        stats[(gender, "ALL")][1] += 1
        used[gender] += 1
        total_done = sum(used.values())
        if total_done % 100 == 0:
            print(f"  evaluated {total_done}", flush=True)

    # ---- report ----
    eths = sorted({e for (_, e) in stats if e != "ALL"})
    print("\n| Group | n | Accuracy |")
    print("|---|---|---|")
    overall = [0, 0]
    for g in ("female", "male"):
        c, n = stats[(g, "ALL")]
        overall[0] += c
        overall[1] += n
        print(f"| {g} (all) | {n} | **{c / max(n, 1):.1%}** |")
        for e in eths:
            c2, n2 = stats[(g, e)]
            if n2:
                print(f"| {g} · {e} | {n2} | {c2 / n2:.1%} |")
    print(f"| **overall** | {overall[1]} | **{overall[0] / max(overall[1], 1):.1%}** |")


if __name__ == "__main__":
    main()
