"""
check_predictions.py — overlay predicted segmentation masks onto their
source images, so you can eyeball inference quality on data/test (which has
no ground-truth masks to score against).

For each image, saves a side-by-side PNG:
    [ original image ]  [ image with predicted mask overlaid ]

Usage:
    python check_predictions.py --images_dir ../data/test/images \
        --masks_dir ../data/test/predictions --output_dir ../data/test/prediction_checks
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from train import CLASS_NAMES, LABEL_VALUES

# One color per raw label value written by inference.py
# (0=background gets no overlay, left as raw image).
LABEL_COLORS = {
    32: (255, 0, 0),  # front_door   - red
    64: (0, 200, 0),  # rear_door    - green
    96: (0, 100, 255),  # front_fender - blue
    128: (255, 200, 0),  # rear_fender  - yellow
    160: (255, 0, 255),  # door_handle  - magenta
}
OVERLAY_ALPHA = 0.45


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    for value, color in LABEL_COLORS.items():
        pixels = mask == value
        if not pixels.any():
            continue
        color_arr = np.array(color, dtype=np.float32)
        result[pixels] = (
            result[pixels] * (1 - OVERLAY_ALPHA) + color_arr * OVERLAY_ALPHA
        )
    return result.astype(np.uint8)


def build_legend(width: int) -> Image.Image:
    legend = Image.new("RGB", (width, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(legend)
    x = 10
    for value, color in LABEL_COLORS.items():
        name = CLASS_NAMES[LABEL_VALUES.index(value)]
        draw.rectangle([x, 10, x + 20, 30], fill=color)
        draw.text((x + 25, 12), name, fill=(0, 0, 0))
        x += 25 + len(name) * 7 + 25
    return legend


def side_by_side(original: Image.Image, overlaid: Image.Image) -> Image.Image:
    target_h = 600
    scale_o = target_h / original.height
    scale_p = target_h / overlaid.height
    original_r = original.resize((int(original.width * scale_o), target_h))
    overlaid_r = overlaid.resize((int(overlaid.width * scale_p), target_h))

    panel = Image.new(
        "RGB",
        (original_r.width + overlaid_r.width + 20, target_h),
        color=(255, 255, 255),
    )
    panel.paste(original_r, (0, 0))
    panel.paste(overlaid_r, (original_r.width + 20, 0))
    return panel


def main():
    parser = argparse.ArgumentParser(
        description="Overlay predicted masks on images for visual inspection"
    )
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument(
        "--masks_dir",
        type=str,
        required=True,
        help="Directory of predicted masks from inference.py "
        "(single-channel PNGs with raw label values, same stem as image).",
    )
    parser.add_argument("--output_dir", type=str, default="prediction_checks")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="If set, only visualize a random sample of this many images "
        "instead of the full directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    pairs = [(p, masks_dir / f"{p.stem}.png") for p in image_paths]
    missing = [img for img, mask in pairs if not mask.exists()]
    pairs = [(img, mask) for img, mask in pairs if mask.exists()]
    if missing:
        print(
            f"Warning: {len(missing)} image(s) had no matching prediction and were skipped."
        )
    if not pairs:
        raise FileNotFoundError(
            f"No matching image/prediction pairs found under {images_dir} / {masks_dir}"
        )

    if args.num_samples:
        rng = random.Random(args.seed)
        pairs = rng.sample(pairs, min(args.num_samples, len(pairs)))

    legend = build_legend(width=1200)

    print(f"Overlaying predictions for {len(pairs)} image(s)")
    for img_path, mask_path in pairs:
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        overlaid = overlay_mask(image, mask)
        panel = side_by_side(Image.fromarray(image), Image.fromarray(overlaid))

        final = Image.new(
            "RGB",
            (panel.width, panel.height + legend.height),
            color=(255, 255, 255),
        )
        final.paste(panel, (0, 0))
        final.paste(
            legend.resize((panel.width, legend.height)), (0, panel.height)
        )

        final.save(output_dir / f"{img_path.stem}_pred_check.png")

    print(f"Saved {len(pairs)} panels to: {output_dir}/")


if __name__ == "__main__":
    main()
