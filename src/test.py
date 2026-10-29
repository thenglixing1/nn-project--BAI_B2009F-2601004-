"""Evaluate a trained model on one data split and print all metrics.

    python src/test.py --model-dir models/final                      # TEST set: final result, run ONCE
    python src/test.py --model-dir models/exp3_aug_long --split val  # safe check on the validation set
    add --plots to also save the confusion matrix and wrong-prediction figures (needs matplotlib)

Writes <split>_metrics.json, <split>_confusion_matrix.csv and
<split>_wrong_predictions.csv into the model folder.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import FoodDataset, load_split
from predict import load_model


def per_class_metrics(cm):
    """cm[i, j] = number of images of true class i that were predicted as class j."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


@torch.no_grad()
def predict_all(model, loader):
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x), dim=1))
    return torch.cat(out)


def majority_baseline(df, classes, split):
    """Accuracy of always predicting the most common training class."""
    train_counts = df[df["split"] == "train"]["cls"].value_counts()
    majority = max(sorted(classes), key=lambda c: train_counts.get(c, 0))
    eval_df = df[df["split"] == split]
    return majority, float((eval_df["cls"] == majority).mean())


def make_plots(cm, classes, wrong, data_dir, fig_dir, split):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image, ImageOps
    except ImportError:
        print("matplotlib not installed: skipping plots.")
        return
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)
    k = len(classes)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    heat = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(k))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(classes)
    for i in range(k):
        for j in range(k):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix ({split} set)")
    fig.colorbar(heat, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{split}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    n = min(10, len(wrong))
    if n:
        fig, axes = plt.subplots(2, 5, figsize=(15, 7))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, row) in zip(axes.flat, wrong.head(n).iterrows()):
            with Image.open(Path(data_dir) / row["rel_path"]) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
            img.thumbnail((300, 300))
            ax.imshow(img)
            ax.set_title(f"true: {row['true']}\npred: {row['predicted']} ({row['confidence']:.0%})",
                         fontsize=9)
        fig.suptitle(f"Most confident wrong predictions ({split} set)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{split}_wrong_predictions.png", dpi=150)
        plt.close(fig)
    print(f"Figures saved to {fig_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/final")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="models/splits.csv")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    if args.split == "test":
        print("NOTE: the test set should be evaluated once, with the final model.\n")

    model, cfg = load_model(args.model_dir)
    classes = cfg["classes"]
    X, y, split_classes = load_split(args.split, args.data_dir, args.splits, cfg["img_size"])
    assert split_classes == classes, "class names in splits.csv do not match the model"

    loader = DataLoader(FoodDataset(X, y, cfg["mean"], cfg["std"], augment=False), batch_size=64)
    probs = predict_all(model, loader)
    pred = probs.argmax(dim=1).numpy()
    true = y.numpy()

    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    precision, recall, f1, support = per_class_metrics(cm)
    accuracy = float((pred == true).mean())

    df = pd.read_csv(args.splits)
    eval_df = df[df["split"] == args.split].sort_values("rel_path").reset_index(drop=True)
    maj_class, maj_acc = majority_baseline(df, classes, args.split)

    print(f"Model: {args.model_dir}   Split: {args.split}   Images: {len(true)}")
    print(f"\nAccuracy: {accuracy:.4f}  ({int((pred == true).sum())}/{len(true)} correct)")
    print(f"\n{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for i, c in enumerate(classes):
        print(f"{c:<12}{precision[i]:>10.3f}{recall[i]:>10.3f}{f1[i]:>10.3f}{int(support[i]):>10d}")
    print(f"{'macro avg':<12}{precision.mean():>10.3f}{recall.mean():>10.3f}{f1.mean():>10.3f}"
          f"{int(support.sum()):>10d}")
    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    cm_df = pd.DataFrame(cm, index=[f"true {c}" for c in classes], columns=classes)
    print(cm_df.to_string())
    print(f"\nSimple baseline (always predict '{maj_class}'): accuracy {maj_acc:.4f}")

    # Wrong predictions, most confident mistakes first
    conf = probs.max(dim=1).values.numpy()
    wrong_idx = np.where(pred != true)[0]
    wrong = pd.DataFrame({
        "rel_path": eval_df["rel_path"].iloc[wrong_idx].values,
        "true": [classes[true[i]] for i in wrong_idx],
        "predicted": [classes[pred[i]] for i in wrong_idx],
        "confidence": np.round(conf[wrong_idx], 3),
    }).sort_values("confidence", ascending=False)

    out_dir = Path(args.model_dir)
    prefix = args.split
    cm_df.to_csv(out_dir / f"{prefix}_confusion_matrix.csv")
    wrong.to_csv(out_dir / f"{prefix}_wrong_predictions.csv", index=False)
    results = {
        "model": str(args.model_dir), "split": args.split, "n_images": int(len(true)),
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "majority_baseline": {"class": maj_class, "accuracy": maj_acc},
    }
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {prefix}_metrics.json, {prefix}_confusion_matrix.csv and "
          f"{prefix}_wrong_predictions.csv to {out_dir}/")

    if args.plots:
        make_plots(cm, classes, wrong, args.data_dir, args.fig_dir, args.split)


if __name__ == "__main__":
    main()