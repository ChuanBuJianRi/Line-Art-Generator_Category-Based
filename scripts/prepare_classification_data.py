#!/usr/bin/env python3
"""
Sample 2000 images per class from iNaturalist / Places365 / CompCars,
split 70% train / 15% val / 15% test, write to dataset_classification/ for ImageFolder.

Splits per class: train 1400, val 300, test 300 (three categories: biological, building, vehicle).
"""

import argparse
import random
import shutil
from pathlib import Path

# Places365 category indices for building-related scenes (from categories_places365.txt)
# Extended list (20 cats) used with val split so we get 2000 images (100 per category).
PLACES365_BUILDING_INDICES = [
    8,   # apartment_building/outdoor
    40,  # barn
    66,  # bridge
    67,  # building_facade
    84,  # castle
    87,  # chalet
    91,  # church/outdoor
    107, # cottage
    113, # dam
    183, # house
    214, # lighthouse
    220, # mansion
    230, # mosque/outdoor
    245, # office_building
    252, # palace
    296, # schoolhouse
    307, # skyscraper
    330, # temple/asia
    334, # tower
    347, # viaduct
]

# 70% train / 15% val / 15% test per class (2000 total per class)
TRAIN_PER_CLASS = 1400   # 70% of 2000
VAL_PER_CLASS = 300      # 15%
TEST_PER_CLASS = 300     # 15%
TOTAL_PER_CLASS = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS  # 2000 per class


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def prepare_biological(
    output_dir: Path,
    inaturalist_root: str | None,
    download: bool,
    seed: int,
    small_download: bool = False,
    cache_dir: str | None = None,
) -> None:
    try:
        import torchvision.datasets as tv_datasets
    except ImportError:
        raise SystemExit("torchvision required: pip install torchvision")

    # Use cache_dir (local disk) if set, else output_dir.parent (e.g. SSD)
    root = Path(inaturalist_root) if inaturalist_root else (Path(cache_dir) / "_cache_inaturalist" if cache_dir else output_dir.parent / "_cache_inaturalist")
    root.mkdir(parents=True, exist_ok=True)

    version = "2021_valid" if small_download else "2021_train_mini"
    print(f"Loading iNaturalist 2021 {version} (biological = animals + humans only, kingdom Animalia)...")
    dataset = tv_datasets.INaturalist(
        str(root),
        version=version,
        download=download,
        target_type="kingdom",
    )
    # Keep only Animalia (animals + humans); exclude Plantae, Fungi, etc.
    animalia_id = None
    for kid in range(20):  # assume at most 20 kingdom ids
        try:
            name = dataset.category_name("kingdom", kid)
            if name == "Animalia":
                animalia_id = kid
                break
        except (KeyError, IndexError, Exception):
            break
    if animalia_id is None:
        raise RuntimeError("Could not find kingdom 'Animalia' in iNaturalist")
    animal_indices = [i for i in range(len(dataset)) if dataset[i][1] == animalia_id]
    n = len(animal_indices)
    if n < TOTAL_PER_CLASS:
        print(f"  Warning: only {n} Animalia images available; using all and splitting.")
        indices = animal_indices
    else:
        rng = random.Random(seed)
        indices = rng.sample(animal_indices, TOTAL_PER_CLASS)

    splits = (
        ("train", indices[:TRAIN_PER_CLASS]),
        ("val", indices[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS]),
        ("test", indices[TRAIN_PER_CLASS + VAL_PER_CLASS :]),
    )
    class_name = "biological"
    for split_name, idx_list in splits:
        if not idx_list:
            continue
        out_dir = output_dir / split_name / class_name
        ensure_dir(out_dir)
        for i, idx in enumerate(idx_list):
            img, _ = dataset[idx]
            out_path = out_dir / f"{i:05d}.jpg"
            if hasattr(img, "save"):
                img.save(out_path)
            else:
                import torch
                if isinstance(img, torch.Tensor):
                    from torchvision.utils import save_image
                    save_image(img, out_path)
                else:
                    raise TypeError("Cannot save image type: " + str(type(img)))
    print(f"  biological written to {output_dir}/<split>/{class_name}, total {len(indices)} images.")


def prepare_building(
    output_dir: Path,
    places365_root: str | None,
    download: bool,
    seed: int,
    small_download: bool = False,
    cache_dir: str | None = None,
) -> None:
    try:
        import torchvision.datasets as tv_datasets
    except ImportError:
        raise SystemExit("torchvision required: pip install torchvision")

    # Use cache_dir (local disk) if set, else output_dir.parent (e.g. SSD)
    root = Path(places365_root) if places365_root else (Path(cache_dir) / "_cache_places365" if cache_dir else output_dir.parent / "_cache_places365")
    root.mkdir(parents=True, exist_ok=True)

    split = "val" if small_download else "train-standard"
    print(f"Loading Places365 {split} small (building categories only)...")
    dataset = tv_datasets.Places365(
        str(root),
        split=split,
        small=True,
        download=download,
    )
    # Filter by .targets to avoid loading every image
    if hasattr(dataset, "targets"):
        building_indices = [i for i, t in enumerate(dataset.targets) if t in PLACES365_BUILDING_INDICES]
    else:
        building_indices = []
        for i in range(len(dataset)):
            _, cat_idx = dataset[i]
            if cat_idx in PLACES365_BUILDING_INDICES:
                building_indices.append(i)
            if len(building_indices) >= TOTAL_PER_CLASS * 2:
                break

    if len(building_indices) < TOTAL_PER_CLASS:
        print(f"  Warning: only {len(building_indices)} building images found; using all.")
    else:
        rng = random.Random(seed)
        building_indices = rng.sample(building_indices, TOTAL_PER_CLASS)

    splits = (
        ("train", building_indices[:TRAIN_PER_CLASS]),
        ("val", building_indices[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS]),
        ("test", building_indices[TRAIN_PER_CLASS + VAL_PER_CLASS :]),
    )
    class_name = "building"
    has_imgs = hasattr(dataset, "imgs")
    for split_name, idx_list in splits:
        if not idx_list:
            continue
        out_dir = output_dir / split_name / class_name
        ensure_dir(out_dir)
        for i, idx in enumerate(idx_list):
            out_path = out_dir / f"{i:05d}.jpg"
            if has_imgs and dataset.imgs[idx][0]:
                src = Path(dataset.imgs[idx][0])
                if src.exists():
                    shutil.copy2(src, out_path)
                    continue
            img, _ = dataset[idx]
            if hasattr(img, "save"):
                img.save(out_path)
            else:
                import torch
                if isinstance(img, torch.Tensor):
                    from torchvision.utils import save_image
                    save_image(img, out_path)
                else:
                    raise TypeError("Cannot save image type: " + str(type(img)))
    print(f"  building written to {output_dir}/<split>/{class_name}, total {len(building_indices)} images.")


def download_stanford_cars(cache_base: Path) -> Path:
    """Download Stanford Cars via HuggingFace (Paulescu/stanford_cars) and save images to cache_base/stanford_cars. Returns that path."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("For --download_vehicle please install: pip install datasets")

    out_dir = cache_base / "stanford_cars"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
    if len(existing) >= 8000:
        print(f"  Stanford Cars cache already has {len(existing)} images, skipping download.")
        return out_dir

    print("  Downloading Stanford Cars from HuggingFace (Paulescu/stanford_cars)...")
    hf_cache = cache_base / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    ds = load_dataset("Paulescu/stanford_cars", cache_dir=str(hf_cache))
    idx = 0
    for split in ds:
        cols = ds[split].column_names
        image_col = next((c for c in ("image", "img", "images") if c in cols), None)
        if image_col is None:
            continue
        n = len(ds[split])
        for i in range(n):
            row = ds[split][i]
            img = row.get(image_col)
            if img is None:
                continue
            path = out_dir / f"{idx:06d}.jpg"
            if hasattr(img, "save"):
                img.save(path)
            else:
                import PIL.Image
                if isinstance(img, PIL.Image.Image):
                    img.save(path)
                else:
                    PIL.Image.fromarray(img).save(path)
            idx += 1
    print(f"  Saved {idx} Stanford Cars images to {out_dir}")
    return out_dir


def prepare_vehicle(output_dir: Path, image_root: str, seed: int) -> None:
    """Sample 2000 vehicle images from a local folder (e.g. CompCars or Stanford Cars)."""
    root_path = Path(image_root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Vehicle image directory not found: {image_root}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_paths = []
    for ext in exts:
        all_paths.extend(root_path.rglob(f"*{ext}"))
    all_paths = [str(p) for p in all_paths]

    if len(all_paths) < TOTAL_PER_CLASS:
        print(f"  Warning: only {len(all_paths)} vehicle images found; using all.")
        chosen = all_paths
    else:
        rng = random.Random(seed)
        chosen = rng.sample(all_paths, TOTAL_PER_CLASS)

    n = len(chosen)
    n_train = min(TRAIN_PER_CLASS, n)
    n_val = min(VAL_PER_CLASS, n - n_train)
    n_test = n - n_train - n_val

    ensure_dir(output_dir / "train" / "vehicle")
    ensure_dir(output_dir / "val" / "vehicle")
    ensure_dir(output_dir / "test" / "vehicle")
    for i, src in enumerate(chosen[:n_train]):
        suf = Path(src).suffix or ".jpg"
        shutil.copy2(src, output_dir / "train" / "vehicle" / f"{i:05d}{suf}")
    for i, src in enumerate(chosen[n_train : n_train + n_val]):
        suf = Path(src).suffix or ".jpg"
        shutil.copy2(src, output_dir / "val" / "vehicle" / f"{i:05d}{suf}")
    for i, src in enumerate(chosen[n_train + n_val :]):
        suf = Path(src).suffix or ".jpg"
        shutil.copy2(src, output_dir / "test" / "vehicle" / f"{i:05d}{suf}")
    print(f"  vehicle written to {output_dir}/<split>/vehicle, total {n} (train {n_train}, val {n_val}, test {n_test}).")


def main():
    # Default: save to project folder (same folder as this script's project root)
    _project_root = Path(__file__).resolve().parent.parent
    _default_output = _project_root / "dataset_classification"
    ap = argparse.ArgumentParser(description="Prepare 3-class data: 2000 per split per class, train/val/test")
    ap.add_argument("--output_dir", type=str, default=str(_default_output), help="Output root (default: project folder/dataset_classification)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--download_bio", action="store_true", help="Download iNaturalist (use --small_download for smaller valid set)")
    ap.add_argument("--download_building", action="store_true", help="Download Places365 (use --small_download for val set only)")
    ap.add_argument("--small_download", action="store_true", help="Use smaller sources: iNaturalist valid (100k), Places365 val (~36k). Download and output both go to output_dir parent (e.g. SSD).")
    ap.add_argument("--inaturalist_root", type=str, default=None, help="Path to extracted iNaturalist (skip download)")
    ap.add_argument("--places365_root", type=str, default=None, help="Path to extracted Places365 (skip download)")
    ap.add_argument("--download_vehicle", action="store_true", help="Download Stanford Cars from HuggingFace and use as vehicle source")
    ap.add_argument("--compcars_root", type=str, default=None, help="CompCars image directory (recursive jpg/png)")
    ap.add_argument("--stanford_cars_root", type=str, default=None, help="Stanford Cars image directory (e.g. from Kaggle; recursive jpg/png)")
    ap.add_argument("--cache_dir", type=str, default=None, help="Dir for download cache (on local disk); only final sampled images go to output_dir. Use when SSD is full.")
    args = ap.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.download_bio or args.inaturalist_root:
        prepare_biological(
            output_dir,
            args.inaturalist_root,
            download=args.download_bio,
            seed=args.seed,
            small_download=args.small_download,
            cache_dir=args.cache_dir,
        )
    else:
        print("Skipping biological (no --download_bio or --inaturalist_root)")

    if args.download_building or args.places365_root:
        prepare_building(
            output_dir,
            args.places365_root,
            download=args.download_building,
            seed=args.seed,
            small_download=args.small_download,
            cache_dir=args.cache_dir,
        )
    else:
        print("Skipping building (no --download_building or --places365_root)")

    vehicle_root = args.compcars_root or args.stanford_cars_root
    if args.download_vehicle and not vehicle_root:
        cache_base = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
        cache_base = cache_base.resolve()
        cache_base.mkdir(parents=True, exist_ok=True)
        vehicle_root = str(download_stanford_cars(cache_base))
    if vehicle_root:
        prepare_vehicle(output_dir, vehicle_root, args.seed)
    else:
        print("Skipping vehicle (no --download_vehicle, --compcars_root or --stanford_cars_root)")

    print("Done. Output dir:", output_dir.resolve())


if __name__ == "__main__":
    main()
