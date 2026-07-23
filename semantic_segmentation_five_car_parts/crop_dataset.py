"""
crop_dataset.py — precompute bounding-box crops for the training set and
save them to disk, so training doesn't have to recompute
estimate_background_color()/compute_foreground_bbox() on every epoch.

Image and mask are cropped to the SAME box (derived from the image only,
never the mask) so they stay pixel-aligned. Filenames are preserved.

Usage:
    python3 crop_dataset.py --images_dir ../data/train/images --masks_dir ../data/train/masks \
        --output_dir ../data/cropped --bbox_threshold 20.0 --bbox_margin 0.03
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image

from bbox_utils import compute_foreground_bbox, estimate_background_color


def main():
    parser = argparse.ArgumentParser(
        description="Precompute bbox crops for images+masks and save to disk"
    )
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/cropped")
    parser.add_argument("--bbox_threshold", type=float, default=20.0)
    parser.add_argument("--bbox_margin", type=float, default=0.03)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_images_dir = Path(args.output_dir) / "images"
    out_masks_dir = Path(args.output_dir) / "masks"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    pairs = [(p, masks_dir / f"{p.stem}.png") for p in image_paths]
    missing = [img for img, mask in pairs if not mask.exists()]
    pairs = [(img, mask) for img, mask in pairs if mask.exists()]
    if missing:
        print(
            f"Warning: {len(missing)} image(s) had no matching mask and were skipped."
        )
    if not pairs:
        raise FileNotFoundError(
            f"No matching image/mask pairs found under {images_dir} / {masks_dir}"
        )

    print(
        f"Cropping {len(pairs)} image/mask pairs "
        f"(threshold={args.bbox_threshold}, margin={args.bbox_margin})"
    )

    for img_path, mask_path in tqdm(pairs):
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        bg_color = estimate_background_color(image)
        top, bottom, left, right = compute_foreground_bbox(
            image, bg_color, args.bbox_threshold, args.bbox_margin
        )

        cropped_image = image[top:bottom, left:right]
        cropped_mask = mask[top:bottom, left:right]

        Image.fromarray(cropped_image).save(
            out_images_dir / f"{img_path.stem}_cropped{img_path.suffix}"
        )
        Image.fromarray(cropped_mask).save(
            out_masks_dir / f"{mask_path.stem}_cropped{mask_path.suffix}"
        )
    print(f"Saved {len(pairs)} cropped image/mask pairs to: {args.output_dir}/")


if __name__ == "__main__":
    main()
