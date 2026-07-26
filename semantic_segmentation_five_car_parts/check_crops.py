"""
check_crops.py — visualize the bounding-box crop on a sample of real
training images for visual inspection, to check whether --bbox_threshold and
--bbox_margin need adjusting before committing to a full training run.

For each sampled image, saves a side-by-side PNG:
    [ original image with bbox drawn ]  [ cropped result ]

Usage:
    python check_crops.py --images_dir data/images --masks_dir data/masks \
        --output_dir crop_checks --num_samples 12
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bbox_utils import (
    compute_foreground_bbox,
    estimate_background_color,
)

LABEL_VALUES = [0, 32, 64, 96, 128, 160]


def draw_bbox(image: np.ndarray, bbox) -> Image.Image:
    top, bottom, left, right = bbox
    pil_img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=6)
    return pil_img


def side_by_side(
    original_with_box: Image.Image, cropped: np.ndarray
) -> Image.Image:
    cropped_img = Image.fromarray(cropped).convert("RGB")

    # Match heights for a clean side-by-side panel.
    target_h = 600
    scale_orig = target_h / original_with_box.height
    scale_crop = target_h / cropped_img.height
    original_resized = original_with_box.resize(
        (int(original_with_box.width * scale_orig), target_h)
    )
    cropped_resized = cropped_img.resize(
        (int(cropped_img.width * scale_crop), target_h)
    )

    panel = Image.new(
        "RGB",
        (original_resized.width + cropped_resized.width + 20, target_h),
        color=(255, 255, 255),
    )
    panel.paste(original_resized, (0, 0))
    panel.paste(cropped_resized, (original_resized.width + 20, 0))
    return panel


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bbox crops on real images"
    )
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help="Optional; only used to also report whether any "
        "labeled part pixels fall outside the crop (a bug signal).",
    )
    parser.add_argument("--output_dir", type=str, default="crop_checks")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--bbox_threshold", type=float, default=20.0)
    parser.add_argument("--bbox_margin", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(images_dir.glob("*.jpg"))
    if not all_images:
        raise FileNotFoundError(f"No .jpg files found in {images_dir}")

    rng = random.Random(args.seed)
    sample = rng.sample(all_images, min(args.num_samples, len(all_images)))

    print(
        f"Checking {len(sample)} of {len(all_images)} images "
        f"(threshold={args.bbox_threshold}, margin={args.bbox_margin})\n"
    )

    flagged = []

    for img_path in sample:
        image = np.array(Image.open(img_path).convert("RGB"))
        bg_color = estimate_background_color(image)
        bbox = compute_foreground_bbox(
            image, bg_color, args.bbox_threshold, args.bbox_margin
        )
        top, bottom, left, right = bbox

        h, w = image.shape[:2]
        box_area_frac = ((bottom - top) * (right - left)) / (h * w)

        # Flag suspicious boxes for manual review.
        note = ""
        if box_area_frac < 0.20:
            note = "SUSPICIOUSLY SMALL crop"
        elif box_area_frac > 0.97:
            note = "SUSPICIOUSLY LARGE crop (background may not be separating)"

        # Optional: check no labeled part pixels fall outside the crop.
        clipped_labels = ""
        if args.masks_dir:
            mask_path = Path(args.masks_dir) / f"{img_path.stem}.png"
            if mask_path.exists():
                mask_raw = np.array(Image.open(mask_path))
                part_mask = mask_raw != 0
                outside = part_mask.copy()
                outside[top:bottom, left:right] = False
                clipped_pixels = int(outside.sum())
                if clipped_pixels > 0:
                    clipped_labels = f"  !! {clipped_pixels} labeled part pixels fall OUTSIDE the crop"

        print(
            f"{img_path.name:30s} bbox_area={box_area_frac:.1%}  {note}{clipped_labels}"
        )
        if note or clipped_labels:
            flagged.append(img_path.name)

        boxed = draw_bbox(image, bbox)
        cropped = image[top:bottom, left:right]
        panel = side_by_side(boxed, cropped)

        # TODO save not only the cropped images but also the corresponding masks
        # for training, because background pixels are not relevant for the segmentation task.
        # This will help in training the model more effectively.

        panel.save(output_dir / f"{img_path.stem}_check.png")

    print(f"\nSaved {len(sample)} comparison panels to: {output_dir}/")
    if flagged:
        print(f"\n{len(flagged)} image(s) flagged for manual review: {flagged}")
    else:
        print("\nNo images flagged — crops look consistent across the sample.")


if __name__ == "__main__":
    main()
