#!/usr/bin/env python3
"""
Organize Sketchy Database into dataset_lineart/<class>/{train,val,test}/{images,targets}/
for U-Net line art generation training.

Categories mapping:
  biological (55 animal categories) -> ~5500 pairs
  building   (7 categories)         -> ~700 pairs  (use multiple sketches to augment)
  vehicle    (12 categories)        -> ~1200 pairs (use multiple sketches to augment)
"""

import argparse
import random
import shutil
from pathlib import Path

BIOLOGICAL_CATS = [
    "ant", "ape", "bat", "bear", "bee", "beetle", "butterfly", "camel", "cat",
    "chicken", "cow", "crab", "crocodilian", "deer", "dog", "dolphin", "duck",
    "elephant", "fish", "frog", "giraffe", "hedgehog", "hermit_crab", "horse",
    "jellyfish", "kangaroo", "lion", "lizard", "lobster", "mouse", "owl",
    "parrot", "penguin", "pig", "rabbit", "raccoon", "ray", "rhinoceros",
    "scorpion", "sea_turtle", "seagull", "seal", "shark", "sheep", "snail",
    "snake", "songbird", "spider", "squirrel", "starfish", "swan", "tiger",
    "turtle", "wading_bird", "zebra",
]

BUILDING_CATS = [
    "cabin", "castle", "church", "skyscraper", "windmill", "door", "window",
]

VEHICLE_CATS = [
    "airplane", "bicycle", "blimp", "car_(sedan)", "helicopter",
    "hot-air_balloon", "motorcycle", "pickup_truck", "rocket", "sailboat",
    "tank", "wheelchair",
]

CLASS_MAP = {}
for c in BIOLOGICAL_CATS:
    CLASS_MAP[c] = "biological"
for c in BUILDING_CATS:
    CLASS_MAP[c] = "building"
for c in VEHICLE_CATS:
    CLASS_MAP[c] = "vehicle"

MAX_SKETCHES_PER_PHOTO = {
    "biological": 1,
    "building": 5,
    "vehicle": 3,
}


def collect_pairs(sketchy_root: Path, max_sketches: dict, seed=42):
    """Collect (photo_path, sketch_path, class_name) triples from Sketchy."""
    photo_dir = sketchy_root / "256x256" / "photo" / "tx_000000000000"
    sketch_dir = sketchy_root / "256x256" / "sketch" / "tx_000000000000"

    pairs = {"biological": [], "building": [], "vehicle": []}
    rng = random.Random(seed)

    for cat_name, cls in CLASS_MAP.items():
        p_dir = photo_dir / cat_name
        s_dir = sketch_dir / cat_name
        if not p_dir.exists() or not s_dir.exists():
            print(f"  Warning: missing {cat_name}, skipping")
            continue

        photos = sorted(p_dir.glob("*.jpg"))
        sketches_all = sorted(s_dir.glob("*.png"))

        sketch_map = {}
        for sk in sketches_all:
            base = sk.stem.rsplit("-", 1)[0]
            sketch_map.setdefault(base, []).append(sk)

        max_sk = max_sketches[cls]

        for photo in photos:
            photo_id = photo.stem
            sk_list = sketch_map.get(photo_id, [])
            if not sk_list:
                continue
            selected = sk_list[:max_sk] if len(sk_list) <= max_sk else rng.sample(sk_list, max_sk)
            for sk in selected:
                pairs[cls].append((photo, sk, cat_name))

    for cls, p in pairs.items():
        rng.shuffle(p)
        print(f"  {cls}: {len(p)} pairs")

    return pairs


def split_and_save(pairs: dict, output_root: Path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split pairs into train/val/test and save to output directory."""
    rng = random.Random(seed)

    for cls, pair_list in pairs.items():
        rng.shuffle(pair_list)
        n = len(pair_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": pair_list[:n_train],
            "val": pair_list[n_train:n_train + n_val],
            "test": pair_list[n_train + n_val:],
        }

        for split_name, split_pairs in splits.items():
            img_dir = output_root / cls / split_name / "images"
            tgt_dir = output_root / cls / split_name / "targets"
            img_dir.mkdir(parents=True, exist_ok=True)
            tgt_dir.mkdir(parents=True, exist_ok=True)

            for idx, (photo, sketch, _) in enumerate(split_pairs):
                shutil.copy2(photo, img_dir / f"{idx:05d}.jpg")
                shutil.copy2(sketch, tgt_dir / f"{idx:05d}.png")

            print(f"  {cls}/{split_name}: {len(split_pairs)} pairs")


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_sketchy = project_root / "cache" / "sketchy" / "extracted"
    default_output = project_root / "dataset_lineart"

    parser = argparse.ArgumentParser(description="Organize Sketchy Database for line art training")
    parser.add_argument("--sketchy_root", type=str, default=str(default_sketchy))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sketchy_root = Path(args.sketchy_root)
    output_root = Path(args.output_dir)

    if output_root.exists():
        print(f"Clearing existing {output_root} ...")
        for cls_dir in output_root.iterdir():
            if cls_dir.is_dir():
                shutil.rmtree(cls_dir)

    print("Collecting photo-sketch pairs ...")
    pairs = collect_pairs(sketchy_root, MAX_SKETCHES_PER_PHOTO, seed=args.seed)

    print("\nSplitting and saving ...")
    split_and_save(pairs, output_root, seed=args.seed)

    print("\nDone!")
    for cls in ["biological", "building", "vehicle"]:
        cls_dir = output_root / cls
        for split in ["train", "val", "test"]:
            img_count = len(list((cls_dir / split / "images").glob("*")))
            print(f"  {cls}/{split}: {img_count} pairs")


if __name__ == "__main__":
    main()
