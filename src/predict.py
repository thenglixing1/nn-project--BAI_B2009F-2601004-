"""Train one model (baseline or experiment) and log the results.

Examples (run from the project root):
    python src/train.py --name baseline --note "Baseline"
    python src/train.py --name exp1_augment --augment --note "Add data augmentation"
    python src/train.py --name exp2_dropout --dropout 0.3 --note "Add dropout 0.3"

Only the train and validation splits are loaded here. The test set is never
touched by this script (it is used once, at the end, by test.py).

Outputs
  models/<name>/model.pt         weights of the epoch with the lowest validation loss
  models/<name>/config.json      image size, mean/std, class names, architecture settings
  models/<name>/history.csv      loss and accuracy per epoch
  models/<name>/architecture.txt layer table with parameter counts
  figures/<name>_curves.png      training vs validation loss and accuracy
  figures/experiment_log.csv     one row per run (for Appendix B of the report)
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FoodDataset, compute_mean_std, load_split
from model import FoodNet, describe_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def plot_curves(history, title, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="validation")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. baseline")
    ap.add_argument("--note", default="", help="what was changed (for the log)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="L2 penalty")
    ap.add_argument("--augment", action="store_true", help="augment training images")
    ap.add_argument("--width", type=int, default=32, help="filters in the first conv layer")
    ap.add_argument("--patience", type=int, default=0, help="early stopping (0 = off)")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="cpu (default, reproducible), mps or cuda")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="models/splits.csv")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.out_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(exist_ok=True)

    print("Loading train and validation images (first run decodes the JPEGs, later runs use the cache)...")
    X_train, y_train, classes = load_split("train", args.data_dir, args.splits, args.img_size)
    X_val, y_val, _ = load_split("val", args.data_dir, args.splits, args.img_size)
    mean, std = compute_mean_std(X_train)  # from the TRAINING set only
    print(f"train {len(y_train)} | val {len(y_val)} | classes {classes}")

    train_ds = FoodDataset(X_train, y_train, mean, std, augment=args.augment)
    val_ds = FoodDataset(X_val, y_val, mean, std, augment=False)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = FoodNet(num_classes=len(classes), width=args.width, dropout=args.dropout).to(device)
    table, n_params = describe_model(model, args.img_size)
    print(table)
    (run_dir / "architecture.txt").write_text(table + "\n")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    history = []
    best = {"val_loss": float("inf")}
    bad_epochs = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f} | {time.time() - t0:.1f}s", flush=True)

        if va_loss < best["val_loss"]:
            best = dict(row)
            torch.save(model.state_dict(), run_dir / "model.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break
    minutes = (time.time() - start) / 60

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    config = {
        "img_size": args.img_size, "mean": mean, "std": std, "classes": classes,
        "width": args.width, "dropout": args.dropout, "seed": args.seed,
        "torch_version": torch.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    plot_curves(history, f"{args.name}: {args.note}", fig_dir / f"{args.name}_curves.png")

    log_path = fig_dir / "experiment_log.csv"
    new_file = not log_path.exists()
    log_row = {
        "exp": args.name, "what_changed": args.note, "epochs_run": len(history),
        "best_epoch": best["epoch"], "lr": args.lr, "batch": args.batch_size,
        "optimizer": args.optimizer, "dropout": args.dropout,
        "weight_decay": args.weight_decay, "augment": args.augment, "width": args.width,
        "train_loss": round(best["train_loss"], 4), "train_acc": round(best["train_acc"], 4),
        "val_loss": round(best["val_loss"], 4), "val_acc": round(best["val_acc"], 4),
        "params": n_params, "minutes": round(minutes, 1),
    }
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(log_row)

    print(f"\nBest epoch {best['epoch']}: val loss {best['val_loss']:.4f}, val acc {best['val_acc']:.3f}")
    print(f"Saved model and config to {run_dir}/, curves to {fig_dir}/{args.name}_curves.png")


if __name__ == "__main__":
    main()