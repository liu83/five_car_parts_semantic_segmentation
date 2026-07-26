"""Compute per-class statistics over the training masks.

Each mask pixel is an integer label from CLASS_MAP. For every class this
reports how many masks contain at least one pixel of that class, and what
fraction of all labeled pixels belong to it.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_MAP = {
    0: "Background",
    32: "Front Door",
    64: "Rear Door",
    96: "Front Fender",
    128: "Rear Fender",
    160: "Door Handle",
}


def compute_statistics(masks_dir: Path) -> dict:
    mask_paths = sorted(masks_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No .png masks found in {masks_dir}")

    image_count = {label: 0 for label in CLASS_MAP}
    pixel_count = {label: 0 for label in CLASS_MAP}
    unknown_labels: set[int] = set()

    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        labels, counts = np.unique(mask, return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            if label not in CLASS_MAP:
                unknown_labels.add(label)
                continue
            image_count[label] += 1
            pixel_count[label] += count

    total_pixels = sum(pixel_count.values())
    return {
        "num_masks": len(mask_paths),
        "image_count": image_count,
        "pixel_count": pixel_count,
        "total_pixels": total_pixels,
        "unknown_labels": unknown_labels,
    }


def print_report(stats: dict) -> None:
    num_masks = stats["num_masks"]
    image_count = stats["image_count"]
    pixel_count = stats["pixel_count"]
    total_pixels = stats["total_pixels"]

    print(f"Total masks scanned: {num_masks}\n")
    header = f"{'Label':>5}  {'Class':<15} {'# Images':>10} {'% Images':>10} {'Pixel Count':>14} {'% Pixels':>10}"
    print(header)
    print("-" * len(header))
    for label, class_name in CLASS_MAP.items():
        n_images = image_count[label]
        n_pixels = pixel_count[label]
        pct_images = 100 * n_images / num_masks if num_masks else 0
        pct_pixels = 100 * n_pixels / total_pixels if total_pixels else 0
        print(
            f"{label:>5}  {class_name:<15} {n_images:>10} {pct_images:>9.1f}% "
            f"{n_pixels:>14} {pct_pixels:>9.1f}%"
        )

    if stats["unknown_labels"]:
        print(
            f"\nWarning: found unexpected label values: {sorted(stats['unknown_labels'])}"
        )
    else:
        stats["unknown_labels"] = None
        print("\nNo unexpected label values found.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--masks_dir",
        type=Path,
        default=Path("data/train/masks"),
        help="Directory containing mask .png files (default: data/train/masks)",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("class_statistics_report.json"),
        help="File to save the statistics report (default: class_statistics_report.json)",
    )
    args = parser.parse_args()

    stats = compute_statistics(args.masks_dir)
    print_report(stats)

    # save stats to a JSON file
    import json

    with open(
        args.output_file,
        "w",
    ) as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
