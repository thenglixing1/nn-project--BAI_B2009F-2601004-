"""Task 2: clean the dataset and create a stratified 70/15/15 split.

Usage (from the project root):
    python src/clean_and_split.py            # dry run: only prints the plan
    python src/clean_and_split.py --apply    # move bad files and write the split

Cleaning rules
  * corrupt / unreadable image            -> moved to data_removed/
  * duplicate appearing in several classes -> all copies moved (label is unknown)
  * duplicate inside one class             -> keep one copy, move the rest

Files are moved, never deleted, so everything can be restored by hand.
The split is saved to models/splits.csv (paths relative to data/) so that the
training and testing containers use exactly the same train/val/test sets.
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

DATA_DIR = Path("data")
REMOVED_DIR = Path("data_removed")
MODELS_DIR = Path("models")
LOG_PATH = Path("figures/cleaning_log.csv")
SPLIT_PATH = MODELS_DIR / "splits.csv"
EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def scan_images():
    rows = []
    for cls_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            ok = True
            try:
                with Image.open(f) as im:
                    im.verify()
            except Exception:
                ok = False
            rows.append({
                "cls": cls_dir.name,
                "path": f,
                "ok": ok,
                "md5": hashlib.md5(f.read_bytes()).hexdigest(),
            })
    return pd.DataFrame(rows)


def plan_removals(df):
    removals = []
    for _, r in df[~df["ok"]].iterrows():
        removals.append({"path": r["path"], "cls": r["cls"], "reason": "corrupt"})

    good = df[df["ok"]]
    for _, g in good.groupby("md5"):
        if len(g) < 2:
            continue
        g = g.sort_values("path")
        if g["cls"].nunique() > 1:
            for _, r in g.iterrows():
                removals.append({"path": r["path"], "cls": r["cls"],
                                 "reason": "duplicate_across_classes"})
        else:
            for _, r in g.iloc[1:].iterrows():
                removals.append({"path": r["path"], "cls": r["cls"],
                                 "reason": "duplicate_same_class"})
    return removals


def stratified_split(df, seed):
    """Split each class separately so class proportions are the same everywhere."""
    rng = random.Random(seed)
    rows = []
    for cls, g in df.groupby("cls"):
        rel = sorted(p.relative_to(DATA_DIR).as_posix() for p in g["path"])
        rng.shuffle(rel)
        n = len(rel)
        n_train = int(n * RATIOS[0])
        n_val = int(n * RATIOS[1])
        for i, r in enumerate(rel):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows.append({"rel_path": r, "cls": cls, "split": split})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually move files and write the split file")
    args = ap.parse_args()

    df = scan_images()
    removals = plan_removals(df)
    rem_paths = {r["path"] for r in removals}

    print(f"Images scanned: {len(df)}")
    print(f"Files to remove: {len(removals)}")
    for r in removals:
        print(f"  {r['reason']:<26} {r['path']}")

    keep = df[~df["path"].isin(rem_paths)]
    split_df = stratified_split(keep, SEED)
    table = pd.crosstab(split_df["cls"], split_df["split"])
    table = table[["train", "val", "test"]]
    print("\nPlanned split (images per class):")
    print(table.to_string())
    print(f"Total kept: {len(keep)}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to make the changes.")
        return

    for r in removals:
        dest = REMOVED_DIR / r["cls"] / r["path"].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(r["path"]), str(dest))
    if removals:
        LOG_PATH.parent.mkdir(exist_ok=True)
        log = pd.DataFrame(removals)
        log["path"] = log["path"].astype(str)
        log.to_csv(LOG_PATH, index=False)
        print(f"\nMoved {len(removals)} files to {REMOVED_DIR}/ (log: {LOG_PATH})")

    MODELS_DIR.mkdir(exist_ok=True)
    split_df.to_csv(SPLIT_PATH, index=False)
    print(f"Split saved to {SPLIT_PATH}")


if __name__ == "__main__":
    main()