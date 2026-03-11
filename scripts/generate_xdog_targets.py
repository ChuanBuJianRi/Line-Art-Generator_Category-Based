#!/usr/bin/env python3
"""
Generate clean line art targets from classification images using
multi-scale edge detection + morphological cleanup.

Output: dataset_lineart/<class>/{train,val,test}/{images,targets}/
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def detect_edges_multiscale(img_gray, sigmas=(1.0, 2.0, 3.0)):
    """Combine edge detection at multiple Gaussian blur scales."""
    arr = np.array(img_gray, dtype=np.float64)
    edges_combined = np.zeros_like(arr)

    for sigma in sigmas:
        radius = max(int(sigma * 2), 1)
        blurred = img_gray.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_arr = np.array(blurred, dtype=np.float64)

        # Sobel-like gradient
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[:, 1:-1] = blurred_arr[:, 2:] - blurred_arr[:, :-2]
        gy[1:-1, :] = blurred_arr[2:, :] - blurred_arr[:-2, :]
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        edges_combined = np.maximum(edges_combined, magnitude)

    return edges_combined


def make_lineart(img, params):
    """
    Generate clean line art from a PIL RGB image.
    Returns a PIL Image (mode 'L') with black lines on white background.
    """
    gray = img.convert("L")

    # Pre-smooth to remove texture noise
    blur_r = params.get("pre_blur", 2)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_r))

    edges = detect_edges_multiscale(gray, sigmas=params["sigmas"])

    if edges.max() > 0:
        edges = edges / edges.max() * 255.0

    # Higher percentile = fewer, stronger edges only
    nonzero = edges[edges > 3]
    if len(nonzero) > 0:
        thresh = np.percentile(nonzero, params["percentile"])
    else:
        thresh = 50

    binary = (edges > thresh).astype(np.uint8) * 255

    # Black lines on white background
    result = 255 - binary
    result_img = Image.fromarray(result, mode="L")

    # Morphological cleanup: remove isolated noise pixels
    result_img = result_img.filter(ImageFilter.MaxFilter(size=3))
    result_img = result_img.filter(ImageFilter.MinFilter(size=3))

    return result_img


CATEGORY_PARAMS = {
    "biological": {
        "sigmas": (1.0, 2.0, 4.0),
        "percentile": 80,
        "pre_blur": 2,
    },
    "building": {
        "sigmas": (1.0, 2.5, 4.0),
        "percentile": 78,
        "pre_blur": 2,
    },
    "vehicle": {
        "sigmas": (1.0, 2.0, 3.5),
        "percentile": 80,
        "pre_blur": 2,
    },
}


def process_category(cls_data_root, output_root, category, params, img_size=256):
    for split in ["train", "val", "test"]:
        src_dir = cls_data_root / split / category
        if not src_dir.exists():
            print(f"  Warning: {src_dir} not found, skipping")
            continue

        img_out = output_root / category / split / "images"
        tgt_out = output_root / category / split / "targets"
        img_out.mkdir(parents=True, exist_ok=True)
        tgt_out.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.*"))
        count = 0
        for i, f in enumerate(files):
            try:
                img = Image.open(f).convert("RGB")
                img = img.resize((img_size, img_size), Image.BILINEAR)

                lineart = make_lineart(img, params)

                img.save(img_out / f"{i:05d}.jpg", quality=90)
                lineart.save(tgt_out / f"{i:05d}.png")
                count += 1
            except Exception as e:
                print(f"  Error: {f.name}: {e}")

        print(f"  {category}/{split}: {count} pairs")


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_cls = project_root / "dataset_classification"
    default_out = project_root / "dataset_lineart"

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_root", type=str, default=str(default_cls))
    parser.add_argument("--output_dir", type=str, default=str(default_out))
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    cls_root = Path(args.cls_root)
    output_root = Path(args.output_dir)

    if output_root.exists():
        print(f"Clearing {output_root} ...")
        for d in output_root.iterdir():
            if d.is_dir():
                shutil.rmtree(d)

    print("Generating line art targets ...\n")
    for cat in ["biological", "building", "vehicle"]:
        print(f"Category: {cat}")
        process_category(cls_root, output_root, cat, CATEGORY_PARAMS[cat], args.img_size)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
