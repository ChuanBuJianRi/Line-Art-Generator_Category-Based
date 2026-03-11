#!/usr/bin/env python3
"""
Flask web app: drag-and-drop an image → classify → generate category-specific line art.
"""

import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS = PROJECT_ROOT / "outputs"

CATEGORIES = ["biological", "building", "vehicle"]
CATEGORY_LABELS = {
    "biological": "Biological",
    "building": "Building",
    "vehicle": "Vehicle",
}

# ── Model definitions (must match training code) ─────────────────────────────

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
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        in_ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

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


# ── Load models ──────────────────────────────────────────────────────────────

device = torch.device("cpu")

print("Loading ResNet-18 classifier ...")
classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load(OUTPUTS / "resnet18" / "best_model.pt", map_location=device, weights_only=True))
classifier.eval()

unet_models = {}
for cat in CATEGORIES:
    print(f"Loading U-Net for {cat} ...")
    m = UNet(in_channels=3, out_channels=1)
    m.load_state_dict(torch.load(OUTPUTS / f"unet_{cat}" / "best_model.pt", map_location=device, weights_only=True))
    m.eval()
    unet_models[cat] = m

print("All models loaded!\n")

# ── Transforms ───────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

cls_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

unet_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="web")


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("web", path)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    # Step 1: Classify
    cls_input = cls_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(cls_input)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    category = CATEGORIES[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs = {CATEGORIES[i]: round(probs[i].item() * 100, 1) for i in range(3)}

    # Step 2: Generate line art
    unet_input = unet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        lineart = unet_models[category](unet_input)

    lineart_np = lineart.squeeze().cpu().numpy()

    # Post-processing: enhance contrast and threshold for clean lines
    import numpy as np
    arr = (lineart_np * 255).astype(np.uint8)

    # Contrast stretch: map [min, max] -> [0, 255]
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi > lo:
        arr = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    # Threshold to clean binary line art (black lines on white)
    threshold = 180
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)

    lineart_img = Image.fromarray(arr, mode="L")

    # Resize to match original aspect ratio (output at 512px max)
    w, h = img.size
    max_dim = 512
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    lineart_img = lineart_img.resize((new_w, new_h), Image.BILINEAR)

    # Encode to base64
    buf = io.BytesIO()
    lineart_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Also encode original (resized) for display
    orig_resized = img.resize((new_w, new_h), Image.BILINEAR)
    buf2 = io.BytesIO()
    orig_resized.save(buf2, format="JPEG", quality=85)
    orig_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    return jsonify({
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "confidence": round(confidence * 100, 1),
        "probabilities": all_probs,
        "lineart": f"data:image/png;base64,{b64}",
        "original": f"data:image/jpeg;base64,{orig_b64}",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
