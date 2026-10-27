"""My own CNN for 5-class Malaysian food classification.

Architecture (width = 32, input 3 x 128 x 128):
  4 x [Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]  with 32, 64, 128, 128 filters
  Global average pooling
  Dense 128 -> 64, ReLU, Dropout
  Dense 64 -> 5   (raw scores; softmax is inside CrossEntropyLoss)

Run `python src/model.py` to print the layer table and parameter counts.
"""
import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FoodNet(nn.Module):
    def __init__(self, num_classes=5, width=32, dropout=0.0):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_block(3, w),
            conv_block(w, 2 * w),
            conv_block(2 * w, 4 * w),
            conv_block(4 * w, 4 * w),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * w, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def describe_model(model, img_size=128):
    """Return (table_text, total_params): layer, type, output shape, parameters."""
    rows, hooks = [], []

    def make_hook(name):
        def hook(module, inputs, output):
            n = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append((name, module.__class__.__name__, tuple(output.shape[1:]), n))
        return hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:  # leaf layers only
            hooks.append(m.register_forward_hook(make_hook(name)))

    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, img_size, img_size, device=device))
    for h in hooks:
        h.remove()
    model.train(was_training)

    line = "-" * 72
    lines = [f"{'Layer':<22}{'Type':<20}{'Output shape':<20}{'Params':>10}", line]
    for name, kind, shape, n in rows:
        lines.append(f"{name:<22}{kind:<20}{str(shape):<20}{n:>10,}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines += [line, f"Total parameters:     {total:,}", f"Trainable parameters: {trainable:,}"]
    return "\n".join(lines), total


if __name__ == "__main__":
    table, _ = describe_model(FoodNet())
    print(table)