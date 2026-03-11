#!/usr/bin/env python3
"""
From dataset_classification, copy images to dataset_lineart/<class>/images/ and
generate pseudo line art targets to dataset_lineart/<class>/targets/ (Canny per class).
Pairing: same numeric id (e.g. 00000.jpg <-> 00000.png).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

CLASSES = ("biological", "building", "vehicle")
# Canny (low, high) per class: biological=softer, building=sharper, vehicle=medium
CANNY_PARAMS = {
    "biological": (40, 120),
    "building": (75, 200),
    "vehicle": (55, 160),
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def collect_class_images(classification_dir: Path, class_name: str) -> list[Path]:
    out = []
    for split in ("train", "val", "test"):
        d = classification_dir / split / class_name
        if not d.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            out.extend(d.glob(ext))
    return sorted(out, key=lambda p: (p.parent.name, p.name))


def _edges_pil_numpy(img_rgb: np.ndarray, strength: float) -> np.ndarray:
    """Fallback: Sobel-like edge magnitude, thresholded. img_rgb: HWC uint8."""
    if img_rgb.ndim == 3:
        gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    else:
        gray = img_rgb.astype(np.float32)
    gx = np.abs(np.diff(gray.astype(np.float32), axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray.astype(np.float32), axis=0, prepend=gray[:1, :]))
    mag = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    thresh = np.percentile(mag, 100 - strength)  # more strength -> more edges
    out = (mag >= thresh).astype(np.uint8) * 255
    return out


def image_to_lineart_canny(img_bgr: np.ndarray, low: int, high: int) -> np.ndarray:
    if HAS_CV2:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, low, high)
    # Fallback without OpenCV: load as RGB and use simple edge strength
    img_rgb = img_bgr[:, :, ::-1] if img_bgr.ndim == 3 else img_bgr
    strength = 15 + (high - low) // 10  # map Canny range to strength
    return _edges_pil_numpy(img_rgb, strength)


def process_class(
    classification_dir: Path,
    lineart_dir: Path,
    class_name: str,
    copy_images: bool,
) -> int:
    paths = collect_class_images(classification_dir, class_name)
    if not paths:
        print(f"  {class_name}: no images found under {classification_dir}/<split>/{class_name}, skip.")
        return 0
    low, high = CANNY_PARAMS.get(class_name, (50, 150))
    images_dir = lineart_dir / class_name / "images"
    targets_dir = lineart_dir / class_name / "targets"
    ensure_dir(images_dir)
    ensure_dir(targets_dir)
    for i, src in enumerate(paths):
        if HAS_CV2:
            img = cv2.imread(str(src))
        else:
            from PIL import Image
            img = np.array(Image.open(src).convert("RGB"))
            if img is None:
                continue
        if img is None or img.size == 0:
            continue
        lineart = image_to_lineart_canny(img, low, high)
        # Save target as 1-channel PNG (0 or 255)
        target_path = targets_dir / f"{i:05d}.png"
        if HAS_CV2:
            cv2.imwrite(str(target_path), lineart)
        else:
            from PIL import Image
            Image.fromarray(lineart).save(target_path)
        if copy_images:
            dest = images_dir / f"{i:05d}{src.suffix}"
            shutil.copy2(src, dest)
    if copy_images:
        print(f"  {class_name}: copied {len(paths)} images -> {images_dir}, generated targets -> {targets_dir} (Canny {low},{high}).")
    else:
        print(f"  {class_name}: generated {len(paths)} targets -> {targets_dir} (Canny {low},{high}).")
    return len(paths)


def main():
    _project_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description="Generate line art targets from dataset_classification and optionally copy images to dataset_lineart."
    )
    ap.add_argument(
        "--classification_dir",
        type=str,
        default=str(_project_root / "dataset_classification"),
        help="Root of classification data (train/val/test/<class>/).",
    )
    ap.add_argument(
        "--lineart_dir",
        type=str,
        default=str(_project_root / "dataset_lineart"),
        help="Root of line art data (output: <class>/images/, <class>/targets/).",
    )
    ap.add_argument(
        "--no_copy_images",
        action="store_true",
        help="Only generate targets; do not copy images to dataset_lineart (assume images already there).",
    )
    ap.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=list(CLASSES),
        help=f"Classes to process. Default: {' '.join(CLASSES)}.",
    )
    args = ap.parse_args()
    classification_dir = Path(args.classification_dir)
    lineart_dir = Path(args.lineart_dir)
    if not classification_dir.is_dir():
        raise FileNotFoundError(f"Classification dir not found: {classification_dir}")
    ensure_dir(lineart_dir)
    copy_images = not args.no_copy_images
    total = 0
    for c in args.categories:
        if c not in CLASSES:
            print(f"  Unknown category '{c}', skip.")
            continue
        n = process_class(classification_dir, lineart_dir, c, copy_images)
        total += n
    print(f"Done. Total pairs: {total}. Line art root: {lineart_dir.resolve()}")


if __name__ == "__main__":
    main()
