"""Download a balanced UTKFace subset and lay it out for train_gender_cnn.py.

Streams the `nu-delta/utkface` dataset from the Hugging Face Hub (no full
1 GB download), takes the first N faces per gender, resizes each to 64x64,
and writes:

    data/train/{female,male}/  (90%)
    data/val/{female,male}/    (10%)

Run:
    python services/prepare_gender_data.py --per-class 2400

Then train:
    python services/train_gender_cnn.py --data data --epochs 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UTKFace subset")
    parser.add_argument("--per-class", type=int, default=2400,
                        help="images per gender class (default 2400)")
    parser.add_argument("--out", default="data", help="output root dir")
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    from datasets import load_dataset

    per_class = args.per_class
    n_val = int(per_class * args.val_frac)

    for split in ("train", "val"):
        for cls in ("female", "male"):
            os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)

    print(f"Streaming nu-delta/utkface: {per_class} per class "
          f"({n_val} val each)...")
    # The hub rows are age-sorted; shuffle the stream (shard + buffer) so the
    # subset spans all ages instead of only age-100 faces.
    ds = load_dataset("nu-delta/utkface", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=4000)

    counts = {"female": 0, "male": 0}
    size = config.GENDER_IMG_SIZE
    for ex in ds:
        gender = str(ex["gender"]).strip().lower()
        if gender not in counts or counts[gender] >= per_class:
            if all(v >= per_class for v in counts.values()):
                break
            continue
        idx = counts[gender]
        split = "val" if idx < n_val else "train"
        img = ex["image"].convert("RGB").resize(size)
        img.save(os.path.join(args.out, split, gender, f"{gender}_{idx:05d}.jpg"),
                 quality=90)
        counts[gender] += 1
        total = sum(counts.values())
        if total % 400 == 0:
            print(f"  {total} saved  {counts}", flush=True)

    print(f"Done: {counts} -> {args.out}/")
    if min(counts.values()) < per_class:
        print("WARNING: stream ended before reaching the requested count.")


if __name__ == "__main__":
    main()
