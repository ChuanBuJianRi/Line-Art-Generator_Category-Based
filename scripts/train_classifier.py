#!/usr/bin/env python3
"""
Train image classifiers (baseline CNN and ResNet-18) on dataset_classification.
Outputs: saved models, training curves (PNG), and a results summary (JSON).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ── Data ──────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def load_datasets(data_root: str):
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_root}/val", transform=val_tf)
    test_ds = datasets.ImageFolder(f"{data_root}/test", transform=val_tf)
    return train_ds, val_ds, test_ds


# ── Models ────────────────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """Simple 3-layer CNN baseline."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18(num_classes=3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def per_class_accuracy(preds, labels, num_classes=3):
    accs = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            accs[c] = 0.0
        else:
            accs[c] = (preds[mask] == c).mean()
    return accs


def confusion_matrix(preds, labels, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Curves saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training(model, model_name, train_ds, val_ds, test_ds, args, device):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    class_names = train_ds.classes
    pca = per_class_accuracy(test_preds, test_labels, num_classes=len(class_names))
    cm = confusion_matrix(test_preds, test_labels, num_classes=len(class_names))

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    for i, name in enumerate(class_names):
        print(f"    {name}: {pca[i]:.4f}")

    # Save plots
    plot_curves(history, str(out_dir / "curves.png"))
    plot_confusion_matrix(cm, class_names, str(out_dir / "confusion_matrix.png"))

    # Save results
    results = {
        "model": model_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "per_class_acc": {class_names[i]: float(pca[i]) for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(),
        "epochs_trained": len(history["train_loss"]),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_dir / 'results.json'}")
    return results


def main():
    _project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(_project_root / "dataset_classification"))
    parser.add_argument("--output_dir", type=str, default=str(_project_root / "outputs"))
    parser.add_argument("--model", type=str, choices=["baseline", "resnet18", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = load_datasets(args.data_root)
    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    all_results = {}

    if args.model in ("baseline", "both"):
        model = BaselineCNN(num_classes=len(train_ds.classes))
        r = run_training(model, "baseline_cnn", train_ds, val_ds, test_ds, args, device)
        all_results["baseline_cnn"] = r

    if args.model in ("resnet18", "both"):
        model = build_resnet18(num_classes=len(train_ds.classes))
        r = run_training(model, "resnet18", train_ds, val_ds, test_ds, args, device)
        all_results["resnet18"] = r

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, r in all_results.items():
        print(f"  {name}: test_acc={r['test_acc']:.4f}, best_val_acc={r['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
