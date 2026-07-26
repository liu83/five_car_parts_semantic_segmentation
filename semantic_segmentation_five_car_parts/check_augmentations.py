"""
check_augmentations.py — visualize augmented (image, mask) pairs for visual inspection.:

For each sampled image, saves one panel PNG:
    [ reference (resized/cropped, no random aug) ] [ aug 1 ] [ aug 2 ] [ aug 3 ] [ aug 4 ]
with the segmentation mask drawn as a colored, semi-transparent overlay.

Usage:
    python check_augmentations.py --images_dir data/images --masks_dir data/masks \
        --output_dir aug_checks --num_images 6 --num_augs 4
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train import CarPartsDataset, CLASS_NAMES, get_transforms

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# One color per class index (0=background gets no overlay, left as raw image).
CLASS_COLORS = {
    1: (255, 0, 0),  # front_door   - red
    2: (0, 200, 0),  # rear_door    - green
    3: (0, 100, 255),  # front_fender - blue
    4: (255, 200, 0),  # rear_fender  - yellow
    5: (255, 0, 255),  # door_handle  - magenta
}
OVERLAY_ALPHA = 0.45


def denormalize(image_tensor) -> np.ndarray:
    img = image_tensor.numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask_tensor) -> np.ndarray:
    mask = mask_tensor.numpy()
    result = image.astype(np.float32).copy()
    for class_idx, color in CLASS_COLORS.items():
        class_pixels = mask == class_idx
        if not class_pixels.any():
            continue
        color_arr = np.array(color, dtype=np.float32)
        result[class_pixels] = (
            result[class_pixels] * (1 - OVERLAY_ALPHA)
            + color_arr * OVERLAY_ALPHA
        )
    return result.astype(np.uint8)


def build_legend(width: int) -> Image.Image:
    legend = Image.new("RGB", (width, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(legend)
    x = 10
    for class_idx, color in CLASS_COLORS.items():
        draw.rectangle([x, 10, x + 20, 30], fill=color)
        draw.text((x + 25, 12), CLASS_NAMES[class_idx], fill=(0, 0, 0))
        x += 25 + len(CLASS_NAMES[class_idx]) * 7 + 25
    return legend


def main():
    parser = argparse.ArgumentParser(
        description="Visualize augmented image/mask pairs"
    )
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="aug_checks")
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument(
        "--num_augs",
        type=int,
        default=4,
        help="Number of randomly-augmented variants to show per image.",
    )
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--pad_to_square", action="store_true")
    parser.add_argument("--no_bbox_crop", action="store_true")
    parser.add_argument("--bbox_threshold", type=float, default=20.0)
    parser.add_argument("--bbox_margin", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(images_dir.glob("*.jpg"))
    pairs = [
        (p, masks_dir / f"{p.stem}.png")
        for p in all_images
        if (masks_dir / f"{p.stem}.png").exists()
    ]
    if not pairs:
        raise FileNotFoundError(
            f"No matching image/mask pairs found under {images_dir} / {masks_dir}"
        )

    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.num_images, len(pairs)))

    crop_to_bbox = not args.no_bbox_crop
    reference_transform = get_transforms(
        train=False, img_size=args.img_size, pad_to_square=args.pad_to_square
    )
    train_transform = get_transforms(
        train=True, img_size=args.img_size, pad_to_square=args.pad_to_square
    )

    reference_ds = CarPartsDataset(
        [p for p, _ in sample],
        [m for _, m in sample],
        reference_transform,
        crop_to_bbox=crop_to_bbox,
        bbox_threshold=args.bbox_threshold,
        bbox_margin=args.bbox_margin,
    )
    train_ds = CarPartsDataset(
        [p for p, _ in sample],
        [m for _, m in sample],
        train_transform,
        crop_to_bbox=crop_to_bbox,
        bbox_threshold=args.bbox_threshold,
        bbox_margin=args.bbox_margin,
    )

    print(
        f"Visualizing {len(sample)} images x {args.num_augs} augmented variants each\n"
    )

    legend = build_legend(width=(args.num_augs + 1) * args.img_size)

    for i, (img_path, _) in enumerate(sample):
        panels = []

        ref_image_t, ref_mask_t = reference_ds[i]
        ref_image = denormalize(ref_image_t)
        ref_overlay = overlay_mask(ref_image, ref_mask_t)
        ref_panel = Image.fromarray(ref_overlay)
        draw = ImageDraw.Draw(ref_panel)
        draw.text((5, 5), "reference", fill=(255, 255, 255))
        panels.append(ref_panel)

        for a in range(args.num_augs):
            aug_image_t, aug_mask_t = train_ds[
                i
            ]  # re-sampling triggers new random augmentation
            aug_image = denormalize(aug_image_t)
            aug_overlay = overlay_mask(aug_image, aug_mask_t)
            aug_panel = Image.fromarray(aug_overlay)
            draw = ImageDraw.Draw(aug_panel)
            draw.text((5, 5), f"aug {a + 1}", fill=(255, 255, 255))
            panels.append(aug_panel)

        row_width = sum(p.width for p in panels) + 10 * (len(panels) - 1)
        row_height = max(p.height for p in panels)
        row = Image.new("RGB", (row_width, row_height), color=(255, 255, 255))
        x_offset = 0
        for p in panels:
            row.paste(p, (x_offset, 0))
            x_offset += p.width + 10

        final = Image.new(
            "RGB",
            (row.width, row.height + legend.height),
            color=(255, 255, 255),
        )
        final.paste(row, (0, 0))
        final.paste(legend.resize((row.width, legend.height)), (0, row.height))

        out_path = output_dir / f"{img_path.stem}_aug_check.png"
        final.save(out_path)
        print(f"Saved: {out_path}")

    print(f"\nAll panels saved to: {output_dir}/")
    print(
        "Check that the colored overlay stays aligned with the actual part "
        "in the image across all variants, and that none of the augmented "
        "versions look unrealistically distorted."
    )


if __name__ == "__main__":
    main()
