#!/usr/bin/env python3
"""
Train category-specific U-Net models for line art generation.
Input: RGB photo (256x256) -> Output: grayscale sketch/line art (256x256).

Trains one U-Net per category (biological, building, vehicle).
Outputs: saved models, loss curves (PNG), sample predictions.
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── Dataset ──────────────────────────────────────────────────────────────────

class LineArtDataset(Dataset):
    """Paired dataset: photo (RGB) -> sketch (grayscale)."""

    def __init__(self, image_dir, target_dir, img_size=256, augment=False):
        self.image_dir = Path(image_dir)
        self.target_dir = Path(target_dir)
        self.img_size = img_size
        self.augment = augment

        self.image_files = sorted(self.image_dir.glob("*.*"))
        self.target_files = sorted(self.target_dir.glob("*.*"))
        assert len(self.image_files) == len(self.target_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.target_files)} targets"

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        tgt = Image.open(self.target_files[idx]).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        tgt = tgt.resize((self.img_size, self.img_size), Image.BILINEAR)

        if self.augment:
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                tgt = tgt.transpose(Image.FLIP_LEFT_RIGHT)

        img = self.to_tensor(img)
        img = self.normalize(img)

        tgt = self.to_tensor(tgt)

        return img, tgt


# ── U-Net Model ──────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net for image-to-image translation (RGB -> grayscale line art)."""

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        in_ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.sigmoid(self.final_conv(x))


# ── Training ─────────────────────────────────────────────────────────────────

class WeightedLineArtLoss(nn.Module):
    """
    Weighted BCE + L1 loss for sparse line art.
    Line pixels (dark) are much rarer than background, so we weight them higher.
    """
    def __init__(self, l1_weight=1.0, bce_weight=1.0, fg_weight=5.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.bce_weight = bce_weight
        self.fg_weight = fg_weight

    def forward(self, pred, target):
        # Weighted BCE: penalize missing lines more
        weight = torch.ones_like(target)
        weight[target < 0.5] = self.fg_weight
        bce = nn.functional.binary_cross_entropy(pred, target, weight=weight)

        l1 = self.l1(pred, target)
        return self.bce_weight * bce + self.l1_weight * l1


def compute_metrics(pred, target, threshold=0.5):
    """Compute Dice coefficient and IoU for binary sketch maps."""
    with torch.no_grad():
        pred_bin = (pred > threshold).float()
        tgt_bin = (target > threshold).float()
        intersection = (pred_bin * tgt_bin).sum()
        union = pred_bin.sum() + tgt_bin.sum()
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        iou = (intersection + 1e-8) / (union - intersection + 1e-8)
    return dice.item(), iou.item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, tgts in loader:
        imgs, tgts = imgs.to(device), tgts.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, tgts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0
    for imgs, tgts in loader:
        imgs, tgts = imgs.to(device), tgts.to(device)
        preds = model(imgs)
        loss = criterion(preds, tgts)
        total_loss += loss.item() * imgs.size(0)
        d, iou = compute_metrics(preds, tgts)
        total_dice += d * imgs.size(0)
        total_iou += iou * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n, total_dice / n, total_iou / n


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_curves(history, save_path, category):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"U-Net Training — {category}", fontsize=14)

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["val_dice"], label="Val Dice", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice Coefficient")
    axes[1].legend()

    axes[2].plot(history["val_iou"], label="Val IoU", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Validation IoU")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Curves saved to {save_path}")


@torch.no_grad()
def save_samples(model, dataset, device, save_path, n=8):
    """Save a grid of input / prediction / ground truth samples."""
    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Prediction")
    axes[0, 2].set_title("Ground Truth")

    indices = list(range(min(n, len(dataset))))

    for row, idx in enumerate(indices):
        img, tgt = dataset[idx]
        inp = img.unsqueeze(0).to(device)
        pred = model(inp).squeeze(0).cpu()

        img_show = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_show = np.clip(img_show, 0, 1)

        axes[row, 0].imshow(img_show)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(pred.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        axes[row, 2].imshow(tgt.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Samples saved to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def train_category(category, data_root, output_dir, args, device):
    print(f"\n{'='*60}")
    print(f"Training U-Net: {category}")
    print(f"{'='*60}")

    cat_root = Path(data_root) / category
    train_ds = LineArtDataset(
        cat_root / "train" / "images", cat_root / "train" / "targets",
        img_size=args.img_size, augment=True)
    val_ds = LineArtDataset(
        cat_root / "val" / "images", cat_root / "val" / "targets",
        img_size=args.img_size)
    test_ds = LineArtDataset(
        cat_root / "test" / "images", cat_root / "test" / "targets",
        img_size=args.img_size)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = WeightedLineArtLoss(l1_weight=1.0, bce_weight=1.0, fg_weight=5.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    out_dir = Path(output_dir) / f"unet_{category}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "val_dice": [], "val_iou": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test — Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")

    plot_curves(history, str(out_dir / "curves.png"), category)
    save_samples(model, test_ds, device, str(out_dir / "samples.png"), n=8)

    results = {
        "category": category,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou,
        "epochs_trained": len(history["train_loss"]),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_dir / 'results.json'}")
    return results


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train U-Net line art generation models")
    parser.add_argument("--data_root", type=str, default=str(project_root / "dataset_lineart"))
    parser.add_argument("--output_dir", type=str, default=str(project_root / "outputs"))
    parser.add_argument("--categories", nargs="+", default=["biological", "building", "vehicle"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    for cat in args.categories:
        r = train_category(cat, args.data_root, args.output_dir, args, device)
        all_results[cat] = r

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for cat, r in all_results.items():
        print(f"  {cat}: test_loss={r['test_loss']:.4f}, dice={r['test_dice']:.4f}, iou={r['test_iou']:.4f}")


if __name__ == "__main__":
    main()
