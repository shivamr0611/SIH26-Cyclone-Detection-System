#!/usr/bin/env python3
"""
augment_dataset.py
Applies data augmentations (flip, ±15 deg rotation, brightness adjustment, Gaussian noise)
to images in database/labeled/ and saves output to database/augmented/.
Uses only OpenCV and NumPy.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Base paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "database" / "labeled"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "database" / "augmented"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by specified angle in degrees around its center."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def adjust_brightness(img: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """Scale pixel intensities by factor (e.g. 1.2 for +20%)."""
    scaled = img.astype(np.float32) * factor
    return np.clip(scaled, 0, 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, sigma: float = 15.0) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_augmentations(img: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate all requested augmentations for an image."""
    return {
        "_flip": cv2.flip(img, 1),
        "_rot15": rotate_image(img, 15.0),
        "_rot-15": rotate_image(img, -15.0),
        "_bright": adjust_brightness(img, 1.2),
        "_noise": add_gaussian_noise(img, sigma=15.0),
    }


def process_dataset(input_dir: Path, output_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Scan category subfolders, apply augmentations, and save to output directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Labeled dataset directory not found: {input_dir}")

    total_source = 0
    total_generated = 0

    # Collect category folders (or process root if no subfolders)
    category_dirs = [p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not category_dirs:
        # Fallback to scanning root of input_dir
        category_dirs = [input_dir]

    for cat_dir in category_dirs:
        rel_category = cat_dir.relative_to(input_dir) if cat_dir != input_dir else Path("")
        target_subfolder = output_dir / rel_category

        image_files = [f for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if not image_files:
            continue

        print(f"\nProcessing category: '{cat_dir.name}' ({len(image_files)} source images)")
        if not dry_run:
            target_subfolder.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            total_source += 1
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARNING] Could not read image: {img_path.name}")
                continue

            stem = img_path.stem
            ext = img_path.suffix
            augs = generate_augmentations(img)

            for suffix, aug_img in augs.items():
                out_name = f"{stem}{suffix}{ext}"
                out_path = target_subfolder / out_name
                total_generated += 1

                if not dry_run:
                    cv2.imwrite(str(out_path), aug_img)

            if dry_run:
                print(f"  [DRY RUN] {img_path.name} -> {len(augs)} variants in {target_subfolder.name}/")

    return total_source, total_generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment cyclone training dataset using OpenCV and NumPy.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Path to labeled input dataset (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to output augmented dataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate augmentation pipeline without writing image files to disk.",
    )
    args = parser.parse_args()

    print("=== Cyclone Dataset Augmentation Pipeline ===")
    print(f"Input:   {args.input_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"Dry run: {args.dry_run}")

    try:
        source_count, gen_count = process_dataset(args.input_dir, args.output_dir, dry_run=args.dry_run)
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")
        return

    print("\n--- Summary ---")
    print(f"Source images scanned:    {source_count}")
    print(f"Augmented files created: {gen_count}")
    if args.dry_run:
        print("[DRY RUN] No files written to disk.")


if __name__ == "__main__":
    main()
