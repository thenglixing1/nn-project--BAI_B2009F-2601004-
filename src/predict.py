"""Load a trained model and predict food classes.

Used by test.py (load_model) and usable on its own for single images:

    python src/predict.py --model-dir models/final path/to/image.jpg [more images...]

Preprocessing is shared with training (dataset.py), so a new image is
transformed exactly like the training data.
"""
import argparse
import json
from pathlib import Path

import torch

from dataset import load_image_uint8, normalise
from model import FoodNet


def load_model(model_dir, device="cpu"):
    """Load models/<name>/config.json and model.pt written by train.py.

    Returns (model in eval mode, cfg dict with classes / img_size / mean / std ...).
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    model = FoodNet(
        num_classes=len(cfg["classes"]),
        width=cfg["width"],
        dropout=cfg["dropout"],
    )
    state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_image(model, cfg, image_path, device="cpu"):
    """Predict one image file. Returns (class name, confidence, {class: probability})."""
    arr = load_image_uint8(image_path, cfg["img_size"])
    x = normalise(arr, cfg["mean"], cfg["std"]).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    best = int(probs.argmax())
    all_probs = {c: float(p) for c, p in zip(cfg["classes"], probs)}
    return cfg["classes"][best], float(probs[best]), all_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="one or more image files")
    ap.add_argument("--model-dir", default="models/final")
    args = ap.parse_args()

    model, cfg = load_model(args.model_dir)
    for path in args.images:
        name, conf, probs = predict_image(model, cfg, path)
        print(f"{path}: {name} ({conf:.1%})")
        for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            print(f"    {c:<12}{p:.3f}")


if __name__ == "__main__":
    main()