"""
inference.py — run the trained car part segmentation model on a directory
of test images and save single-channel prediction masks.

Usage:
    python3 inference.py --input path/to/test_images --output path/to/predictions

By default, preprocessing settings (img_size, pad_to_square, bbox crop
threshold/margin) are read automatically from the config.yaml saved next to
best_model.pth during training -- this guarantees inference uses EXACTLY the
same preprocessing as training, which matters a lot here: a mismatch (e.g.
training with pad_to_square=True but inferring with plain resize) would
silently distort part geometry and tank accuracy without any error message.

Preprocessing pipeline per image (mirrors train.py's val/test transform):
    1. Crop to the detected car bounding box (crop_to_object) -- removes the
       large background canvas so resolution is spent on the car itself.
    2. Resize to img_size x img_size, either by:
         - plain stretch (pad_to_square=False), or
         - letterbox: scale longest side to img_size, pad the rest with
           background (pad_to_square=True) -- reimplemented manually here
           with cv2 (rather than calling albumentations) so the exact scale
           and padding offsets are known and can be inverted precisely.
    3. Normalize with ImageNet mean/std, run through the model.
    4. Invert step 2 (crop out padding if letterboxed, resize prediction
       back to the crop's pixel dimensions using NEAREST interpolation --
       critical, since bilinear would blend class indices into invalid
       intermediate values).
    5. Paste the crop-sized prediction back onto a full-size canvas matching
       the original image dimensions, with everything outside the crop set
       to background (class 0) -- correct, since that region genuinely is
       background canvas.
    6. Map class indices (0..5) back to the raw label values the assignment
       specifies (0, 32, 64, 96, 128, 160) and save as a single-channel PNG.
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from PIL import Image

from bbox_utils import crop_to_object
from train import CLASS_NAMES, INDEX_TO_VALUE, LABEL_VALUES, NUM_CLASSES

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fallback preprocessing defaults, used ONLY if config.yaml can't be found.
# These match train.py's argparse defaults -- but relying on this fallback
# is a red flag: it means training/inference are no longer guaranteed to
# match, so a loud warning is printed if it's ever used.
FALLBACK_DEFAULTS = {
    "img_size": 512,
    "pad_to_square": False,
    "no_bbox_crop": False,
    "bbox_threshold": 20.0,
    "bbox_margin": 0.03,
}


# train.py timestamps each run's config as config_<YYYY-MM-DD_HH-MM-SS>.yaml.
CONFIG_TIMESTAMP_RE = re.compile(
    r"config_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.yaml$"
)


def report_path_with_config_timestamp(
    report_path: Path, config_path: Path
) -> Path:
    """Append the timestamp embedded in config_path's filename to
    report_path, so a report can always be traced back to the exact run
    config it was generated with (e.g. inference_report_2026-07-23_16-16-00.yaml).
    """
    match = CONFIG_TIMESTAMP_RE.search(config_path.name)
    if not match:
        return report_path
    timestamp = match.group(1)
    return report_path.with_name(
        f"{report_path.stem}_{timestamp}{report_path.suffix}"
    )


def load_preprocessing_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        hp = config["hyperparameters"]
        print(f"Loaded preprocessing config from: {config_path}")
        return {
            "img_size": hp["img_size"],
            "pad_to_square": hp["pad_to_square"],
            "crop_to_bbox": not hp["no_bbox_crop"],
            "bbox_threshold": hp["bbox_threshold"],
            "bbox_margin": hp["bbox_margin"],
        }
    else:
        print(
            f"WARNING: config.yaml not found at {config_path}. "
            f"Falling back to default preprocessing settings -- these MUST "
            f"match how the model was trained, or predictions will be wrong "
            f"without any error being raised. Falling back to: {FALLBACK_DEFAULTS}"
        )
        return {
            "img_size": FALLBACK_DEFAULTS["img_size"],
            "pad_to_square": FALLBACK_DEFAULTS["pad_to_square"],
            "crop_to_bbox": not FALLBACK_DEFAULTS["no_bbox_crop"],
            "bbox_threshold": FALLBACK_DEFAULTS["bbox_threshold"],
            "bbox_margin": FALLBACK_DEFAULTS["bbox_margin"],
        }


def resize_for_inference(image: np.ndarray, img_size: int, pad_to_square: bool):
    """Deterministic resize matching train.py's val/test transform.
    Returns (model_input_image, inverse_meta) where inverse_meta carries
    everything needed to map a prediction back to `image`'s original shape.
    """
    h, w = image.shape[:2]

    if not pad_to_square:
        resized = cv2.resize(
            image, (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )
        meta = {"mode": "resize", "orig_h": h, "orig_w": w}
        return resized, meta

    # Letterbox: scale longest side to img_size, then center-pad.
    scale = img_size / max(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = img_size - new_h, img_size - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(
        scaled,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    meta = {
        "mode": "pad",
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return padded, meta


def restore_prediction(pred_mask: np.ndarray, meta: dict) -> np.ndarray:
    """Invert resize_for_inference: map a (img_size, img_size) class-index
    prediction back to the original (pre-resize) crop dimensions."""
    if meta["mode"] == "resize":
        return cv2.resize(
            pred_mask,
            (meta["orig_w"], meta["orig_h"]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Undo padding first, then undo the aspect-preserving scale.
    top, left = meta["pad_top"], meta["pad_left"]
    new_h, new_w = meta["new_h"], meta["new_w"]
    unpadded = pred_mask[top : top + new_h, left : left + new_w]
    return cv2.resize(
        unpadded,
        (meta["orig_w"], meta["orig_h"]),
        interpolation=cv2.INTER_NEAREST,
    )


def load_model(model_path: Path, device: torch.device):
    state_dict = torch.load(model_path, map_location=device)
    is_fp16 = next(iter(state_dict.values())).dtype == torch.float16

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # loading trained weights below, no need for ImageNet init
        in_channels=3,
        classes=NUM_CLASSES,
    )
    if is_fp16:
        model = model.half()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(
        f"Loaded model from {model_path} ({'fp16' if is_fp16 else 'fp32'} weights)"
    )
    return model, is_fp16


@torch.no_grad()
def run_inference(
    model, image: np.ndarray, preprocess_cfg: dict, device, is_fp16: bool
):
    """Full pipeline for one image: crop -> resize -> predict -> restore.
    Returns a class-index mask matching the ORIGINAL image's H x W."""
    orig_h, orig_w = image.shape[:2]

    if preprocess_cfg["crop_to_bbox"]:
        cropped, _, bbox = crop_to_object(
            image,
            threshold=preprocess_cfg["bbox_threshold"],
            margin_frac=preprocess_cfg["bbox_margin"],
        )
    else:
        cropped, bbox = image, (0, orig_h, 0, orig_w)

    model_input, resize_meta = resize_for_inference(
        cropped, preprocess_cfg["img_size"], preprocess_cfg["pad_to_square"]
    )

    normalized = (model_input.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = (
        torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    )
    if is_fp16:
        tensor = tensor.half()

    logits = model(tensor)
    pred_indices = (
        torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    )

    # Map the (img_size, img_size) prediction back to the crop's original
    # pixel dimensions, then paste onto a full-size canvas (background
    # everywhere outside the crop -- that region genuinely IS background).
    crop_pred = restore_prediction(pred_indices, resize_meta)

    full_pred = np.zeros((orig_h, orig_w), dtype=np.uint8)
    top, bottom, left, right = bbox
    full_pred[top:bottom, left:right] = crop_pred

    return full_pred


def indices_to_label_values(pred_indices: np.ndarray) -> np.ndarray:
    """Map internal class indices (0..5) back to the raw label values the
    assignment specifies (0, 32, 64, 96, 128, 160)."""
    label_mask = np.zeros_like(pred_indices, dtype=np.uint8)
    for index, value in INDEX_TO_VALUE.items():
        label_mask[pred_indices == index] = value
    return label_mask


def save_inference_report(
    report_path: Path,
    args,
    device,
    model_path: Path,
    config_path: Path,
    preprocess_cfg: dict,
    num_images: int,
    forward_times: list,
    total_times: list,
):
    """Save a YAML report covering: exact preprocessing config used, model
    file size, hardware, and profiling stats -- checked explicitly against
    the assignment's constraints (model size < 180MB, inference < 1s/image
    on the reference RTX 5090). Keeping this separate from predictions/
    keeps that folder containing only the deliverable masks, per the spec.
    """
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    avg_forward_s = sum(forward_times) / len(forward_times)
    avg_total_s = sum(total_times) / len(total_times)

    is_reference_gpu = (
        device.type == "cuda" and "5090" in torch.cuda.get_device_name(0)
    )

    report = {
        "run_info": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_dir": str(args.input),
            "output_dir": str(args.output_dir),
            "model_path": str(model_path),
            "config_path_used": (
                str(config_path) if config_path.exists() else None
            ),
            "num_images_processed": num_images,
        },
        "hardware": {
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "torch_version": str(torch.__version__),
            "is_assignment_reference_gpu_rtx5090": is_reference_gpu,
        },
        "model": {
            "architecture": "Unet",
            "encoder": "resnet34",
            "file_size_mb": round(model_size_mb, 2),
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "label_values": LABEL_VALUES,
        },
        "preprocessing_used": preprocess_cfg,
        "profiling": {
            "avg_forward_ms": round(avg_forward_s * 1000, 2),
            "avg_total_ms_incl_io": round(avg_total_s * 1000, 2),
            "min_forward_ms": round(min(forward_times) * 1000, 2),
            "max_forward_ms": round(max(forward_times) * 1000, 2),
        },
        "constraint_checks": {
            "model_size_under_180mb": model_size_mb < 180,
            "avg_forward_time_under_1s": avg_forward_s < 1.0,
            "note": (
                "Timed on the device listed under 'hardware' above, not the "
                "assignment's RTX 5090 reference GPU -- see is_assignment_reference_gpu_rtx5090."
                if not is_reference_gpu
                else "Timed directly on the assignment's RTX 5090 reference GPU."
            ),
        },
    }

    with open(report_path, "w") as f:
        yaml.safe_dump(report, f, sort_keys=False, default_flow_style=False)

    print(f"\nInference report saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run car part segmentation inference"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Directory of input images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save predicted masks.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/training_outputs/best_model.pth",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the training config.yaml. Defaults to "
        "config.yaml next to --model_path.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="results/inference_outputs/inference_report.yaml",
        help="Where to save the YAML report (profiling + config + "
        "constraint checks). Kept outside --output so the "
        "predictions/ folder only contains deliverable masks.",
    )
    args = parser.parse_args()

    datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    input_dir = Path(args.input)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    config_path = (
        Path(args.config_path)
        if args.config_path
        else model_path.parent / "config.yaml"
    )
    preprocess_cfg = load_preprocessing_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, is_fp16 = load_model(model_path, device)

    image_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"Found {len(image_paths)} images to process.")

    forward_times = []
    total_times = []

    for image_path in image_paths:
        total_start = time.time()

        image = np.array(Image.open(image_path).convert("RGB"))

        forward_start = time.time()
        pred_indices = run_inference(
            model, image, preprocess_cfg, device, is_fp16
        )
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure GPU work is finished before timing
        forward_times.append(time.time() - forward_start)

        label_mask = indices_to_label_values(pred_indices)
        out_path = output_dir / f"{image_path.stem}.png"
        Image.fromarray(label_mask, mode="L").save(out_path)

        total_times.append(time.time() - total_start)

    avg_forward = sum(forward_times) / len(forward_times)
    avg_total = sum(total_times) / len(total_times)

    print(f"\nProcessed {len(image_paths)} images.")
    print(f"Avg model inference time per image: {avg_forward * 1000:.1f} ms")
    print(f"Avg total time per image (incl. I/O): {avg_total * 1000:.1f} ms")
    if device.type == "cuda":
        print(
            f"Hardware: {torch.cuda.get_device_name(0)} "
            f"(assignment reference hardware: NVIDIA RTX 5090 -- "
            f"note any difference in your README if not using that GPU)."
        )
    print(f"Predictions saved to: {output_dir}/")

    report_path = report_path_with_config_timestamp(
        Path(args.report_path), config_path
    )
    report = save_inference_report(
        report_path,
        args,
        device,
        model_path,
        config_path,
        preprocess_cfg,
        len(image_paths),
        forward_times,
        total_times,
    )
    checks = report["constraint_checks"]
    print(
        f"Constraint check -- model size < 180MB: {checks['model_size_under_180mb']} "
        f"({report['model']['file_size_mb']} MB)"
    )
    print(
        f"Constraint check -- inference < 1s/image: {checks['avg_forward_time_under_1s']} "
        f"({report['profiling']['avg_forward_ms']} ms)"
    )


if __name__ == "__main__":
    main()
