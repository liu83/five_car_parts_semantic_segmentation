"""
bbox_utils.py — shared utilities for cropping car images to the object's
bounding box before resizing.

Background removal in the dataset means each image has a large
surrounding canvas of near-uniform background color, with the car occupying
only part of the frame. Cropping to the car's bounding box before resizing
increases the effective resolution of small parts (e.g. Door Handle)
"for free" — no extra compute, since we still resize to the same final
img_size afterward.
"""

import numpy as np


def estimate_background_color(
    image: np.ndarray, border_frac: float = 0.02
) -> np.ndarray:
    """Estimate background color from a thin border strip around the image,
    using the per-channel median (robust to a few stray foreground pixels
    that might touch the edge)."""
    h, w = image.shape[:2]
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    border_pixels = np.concatenate(
        [
            image[:bh, :, :].reshape(-1, image.shape[2]),
            image[-bh:, :, :].reshape(-1, image.shape[2]),
            image[:, :bw, :].reshape(-1, image.shape[2]),
            image[:, -bw:, :].reshape(-1, image.shape[2]),
        ],
        axis=0,
    )

    return np.median(border_pixels, axis=0)


def compute_foreground_bbox(
    image: np.ndarray,
    bg_color: np.ndarray,
    threshold: float = 20.0,
    margin_frac: float = 0.03,
) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) bounding box of pixels that differ
    from the estimated background color by more than `threshold` (summed
    absolute per-channel difference), expanded by a small margin.

    Falls back to the full image if no foreground is detected, so a bad
    estimate never corrupts a sample — it just skips the crop for that image.
    """
    diff = np.abs(image.astype(np.float32) - bg_color.astype(np.float32)).sum(
        axis=2
    )
    foreground = diff > threshold

    rows = np.any(foreground, axis=1)
    cols = np.any(foreground, axis=0)

    h, w = image.shape[:2]
    if not rows.any() or not cols.any():
        return 0, h, 0, w  # fallback: no crop

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    margin_h = int((bottom - top) * margin_frac)
    margin_w = int((right - left) * margin_frac)

    top = max(0, int(top) - margin_h)
    bottom = min(h, int(bottom) + margin_h + 1)
    left = max(0, int(left) - margin_w)
    right = min(w, int(right) + margin_w + 1)

    return top, bottom, left, right


def crop_to_object(
    image: np.ndarray,
    mask: np.ndarray = None,
    threshold: float = 20.0,
    margin_frac: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop `image` (and optionally `mask`, using the same box) to the
    detected object bounding box.

    Returns:
        cropped_image, cropped_mask_or_None, bbox (top, bottom, left, right)
        — the bbox is in ORIGINAL image coordinates, which inference.py
        needs to map predictions back onto the full-size output canvas.
    """
    bg_color = estimate_background_color(image)
    bbox = compute_foreground_bbox(image, bg_color, threshold, margin_frac)
    top, bottom, left, right = bbox

    cropped_image = image[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right] if mask is not None else None

    return cropped_image, cropped_mask, bbox
