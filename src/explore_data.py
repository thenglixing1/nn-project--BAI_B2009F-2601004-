"""Task 2: explore the Malaysian food image dataset.

Run from the project root:
    python src/explore_data.py

Expects data/<class_name>/*.jpg and writes 3 figures into figures/.
"""
import hashlib
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
SAMPLES_PER_CLASS = 5


def scan_images():
    """Return one row per image: class, path, validity, size, mode, md5 hash."""
    rows = []
    class_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    for cls_dir in class_dirs:
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            row = {
                "cls": cls_dir.name,
                "path": str(f),
                "ok": True,
                "w": None,
                "h": None,
                "mode": None,
                "md5": hashlib.md5(f.read_bytes()).hexdigest(),
            }
            try:
                with Image.open(f) as im:
                    im.verify()  # detects truncated / corrupt files
                with Image.open(f) as im:  # verify() closes the file, reopen
                    row["w"], row["h"], row["mode"] = im.width, im.height, im.mode
            except Exception:
                row["ok"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df):
    print("\n=== Images per class ===")
    print(df.groupby("cls").size().to_string())
    print(f"\nTotal images: {len(df)}")

    bad = df[~df["ok"]]
    print(f"\n=== Corrupt / unreadable images: {len(bad)} ===")
    for p in bad["path"].head(10):
        print("  ", p)

    good = df[df["ok"]]
    print("\n=== Colour modes ===")
    print(good["mode"].value_counts().to_string())

    print("\n=== Image size (pixels) ===")
    print(good[["w", "h"]].describe().round(1).to_string())

    dup = df[df.duplicated("md5", keep=False)]
    n_groups = dup["md5"].nunique()
    cross = dup.groupby("md5")["cls"].nunique()
    n_cross = int((cross > 1).sum())
    print(f"\n=== Exact duplicates: {len(dup)} files in {n_groups} groups ===")
    print(f"Groups that appear in more than one class: {n_cross}")


def plot_class_distribution(df):
    counts = df.groupby("cls").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.bar_label(bars)
    ax.set_title("Figure 1: Number of images per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_class_distribution.png", dpi=150)
    plt.close(fig)


def plot_samples(df):
    rng = random.Random(SEED)
    classes = sorted(df["cls"].unique())
    fig, axes = plt.subplots(
        len(classes), SAMPLES_PER_CLASS,
        figsize=(2.2 * SAMPLES_PER_CLASS, 2.2 * len(classes)),
    )
    for r, cls in enumerate(classes):
        paths = df[(df["cls"] == cls) & df["ok"]]["path"].tolist()
        picks = rng.sample(paths, min(SAMPLES_PER_CLASS, len(paths)))
        for c in range(SAMPLES_PER_CLASS):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(picks):
                with Image.open(picks[c]) as im:
                    ax.imshow(im.convert("RGB"))
            if c == 0:
                ax.set_title(cls, loc="left", fontsize=10)
    fig.suptitle("Figure 2: Random sample images per class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_samples.png", dpi=120)
    plt.close(fig)


def plot_sizes(df):
    good = df[df["ok"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, g in good.groupby("cls"):
        ax.scatter(g["w"], g["h"], s=8, alpha=0.4, label=cls)
    ax.set_title("Figure 3: Image width vs height")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_image_sizes.png", dpi=150)
    plt.close(fig)


def main():
    if not DATA_DIR.exists():
        raise SystemExit("data/ folder not found. Run this from the project root.")
    FIG_DIR.mkdir(exist_ok=True)

    df = scan_images()
    if df.empty:
        raise SystemExit("No images found. Expected data/<class_name>/*.jpg")

    print_summary(df)
    plot_class_distribution(df)
    plot_samples(df)
    plot_sizes(df)
    df.drop(columns=["md5"]).to_csv(FIG_DIR / "image_index.csv", index=False)
    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()