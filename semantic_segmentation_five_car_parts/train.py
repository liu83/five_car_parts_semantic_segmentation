"""
train.py — Car Part Semantic Segmentation
Architecture: U-Net (SMP) with ResNet34 encoder (ImageNet pretrained)
Classes: 0=Background, 32=Front Door, 64=Rear Door, 96=Front Fender,
         128=Rear Fender, 160=Door Handle

Usage:
    python3 train.py --images_dir data/images --masks_dir data/masks \
        --output_dir outputs --epochs 60 --img_size 384

Design notes (see README for full justification):
    - 90/10 train/val split, seeded for reproducibility.
    - Encoder frozen for the first `--freeze_epochs` epochs (train decoder only),
      then unfrozen and fine-tuned end-to-end with a lower encoder LR.
    - Loss = 0.5 * Dice + 0.5 * class-weighted CrossEntropy, to address the
      severe class imbalance (Door Handle ~0.4% of pixels vs Background ~67%).
    - Class weights computed automatically from the TRAINING split masks only
      (no validation leakage) using log-scaled inverse frequency (ENet-style).
    - Mixed precision (AMP) used throughout for training speed and to keep
      inference-time behavior consistent with fp16 export.
    - Best checkpoint selected by validation mIoU, with early stopping.
"""

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import yaml

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from bbox_utils import crop_to_object

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_VALUES = [0, 32, 64, 96, 128, 160]  # raw pixel values in mask PNGs
CLASS_NAMES = [
    "background",
    "front_door",
    "rear_door",
    "front_fender",
    "rear_fender",
    "door_handle",
]
NUM_CLASSES = len(LABEL_VALUES)
VALUE_TO_INDEX = {v: i for i, v in enumerate(LABEL_VALUES)}
INDEX_TO_VALUE = {i: v for i, v in enumerate(LABEL_VALUES)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CarPartsDataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        transform,
        crop_to_bbox: bool = True,
        bbox_threshold: float = 20.0,
        bbox_margin: float = 0.03,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.crop_to_bbox = crop_to_bbox
        self.bbox_threshold = bbox_threshold
        self.bbox_margin = bbox_margin

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_raw = np.array(Image.open(self.mask_paths[idx]))

        # Map raw label values (0,32,64,...) -> class indices (0..5)
        mask = np.zeros_like(mask_raw, dtype=np.uint8)
        for value, index in VALUE_TO_INDEX.items():
            mask[mask_raw == value] = index

        # Crop out the large surrounding background canvas BEFORE resizing.
        # This concentrates resolution on the car itself (e.g. more pixels
        # per Door Handle) instead of wasting them on empty background.
        # Same function is used in inference.py so train/test preprocessing
        # stays identical.
        if self.crop_to_bbox:
            image, mask, _ = crop_to_object(
                image,
                mask,
                threshold=self.bbox_threshold,
                margin_frac=self.bbox_margin,
            )

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].long()
        return image_t, mask_t


def get_resize_block(img_size: int, pad_to_square: bool):
    """Two supported resize strategies, chosen via --pad_to_square:

    1. Plain Resize(img_size, img_size) [default]: stretches to a square.
       Fine here since crops are already close to square (~1.09 aspect
       ratio for the raw 3000x3264 images), so distortion is mild.

    2. Letterbox (LongestMaxSize + PadIfNeeded): preserves aspect ratio
       exactly by scaling the longest side to img_size and padding the
       rest with background (value=0, i.e. class index 0 on the mask,
       which is semantically correct since it IS background). Avoids any
       shape distortion at the cost of some wasted padding pixels.
    """
    if pad_to_square:
        return [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
        ]
    return [A.Resize(img_size, img_size)]


def get_transforms(train: bool, img_size: int, pad_to_square: bool = False):
    mean = (0.485, 0.456, 0.406)  # ImageNet stats, matches pretrained encoder
    std = (0.229, 0.224, 0.225)
    resize_block = get_resize_block(img_size, pad_to_square)

    if train:
        return A.Compose(
            resize_block
            + [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.4
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.3,
                ),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            resize_block
            + [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def split_dataset(
    images_dir: Path, masks_dir: Path, val_fraction: float, seed: int
):
    # TODO consider splitting train and val dataset based on class distribution,
    # to ensure that both splits have a similar distribution of classes.
    # This can help in training a more robust model and avoid overfitting to
    # a particular class distribution in the training set.

    image_files = sorted(images_dir.glob("*.jpg"))
    pairs = []
    for img_path in image_files:
        mask_path = masks_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"Warning: no mask found for {img_path.name}, skipping.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_images, train_masks = zip(*train_pairs)
    val_images, val_masks = zip(*val_pairs)
    return (
        list(train_images),
        list(train_masks),
        list(val_images),
        list(val_masks),
    )


# ---------------------------------------------------------------------------
# Class weights (computed from TRAIN split masks only)
# ---------------------------------------------------------------------------
def compute_class_weights(mask_paths, num_classes=NUM_CLASSES):
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for mask_path in mask_paths:
        mask_raw = np.array(Image.open(mask_path))
        for value, index in VALUE_TO_INDEX.items():
            pixel_counts[index] += np.sum(mask_raw == value)

    total = pixel_counts.sum()
    freqs = pixel_counts / total

    # ENet-style log-scaled inverse frequency weighting.
    # Keeps rare-class weights (e.g. door_handle) bounded rather than exploding,
    # which would otherwise destabilize training.
    weights = 1.0 / np.log(1.02 + freqs)
    weights = weights / weights.mean()  # normalize around 1.0

    print("Class pixel frequencies and computed CE weights:")
    for name, freq, weight in zip(CLASS_NAMES, freqs, weights):
        print(f"  {name:15s} freq={freq:.4%}  weight={weight:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class DiceCELoss(nn.Module):
    """0.5 * multiclass Dice + 0.5 * weighted CrossEntropy."""

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_iou_per_class(logits, targets, num_classes=NUM_CLASSES, eps=1e-7):
    preds = torch.argmax(logits, dim=1)
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float("nan"))  # class absent in this batch
        else:
            ious.append(intersection / (union + eps))
    return ious


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    class_iou_sums = np.zeros(NUM_CLASSES)
    class_iou_counts = np.zeros(NUM_CLASSES)

    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(
            device, non_blocking=True
        )
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)

        ious = compute_iou_per_class(logits, masks)
        for c, iou in enumerate(ious):
            if not math.isnan(iou):
                class_iou_sums[c] += iou
                class_iou_counts[c] += 1

    val_loss = running_loss / len(loader.dataset)
    per_class_iou = class_iou_sums / np.maximum(class_iou_counts, 1)
    mean_iou = per_class_iou.mean()
    return val_loss, mean_iou, per_class_iou


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(
            1, total_epochs - warmup_epochs
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def save_run_config(
    args,
    device,
    output_dir: Path,
    datetime_stamp: str,
    num_train: int,
    num_val: int,
):
    """Save every training hyperparameter plus key environment/reproducibility
    details to a YAML file, so a specific run's exact configuration is always
    recoverable later -- not just implicitly encoded in whatever CLI command
    happened to be typed.
    """
    config = {
        "run_info": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "torch_version": str(torch.__version__),
            "num_train_images": num_train,
            "num_val_images": num_val,
        },
        "model": {
            "architecture": "Unet",
            "encoder": "resnet34",
            "encoder_weights": "imagenet",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "label_values": LABEL_VALUES,
        },
        "hyperparameters": vars(args),
    }

    config_path = output_dir / f"config_{datetime_stamp}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"Run configuration saved to: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Train car part segmentation model"
    )
    parser.add_argument("--images_dir", type=str, default="data/images")
    parser.add_argument("--masks_dir", type=str, default="data/masks")
    parser.add_argument(
        "--output_dir", type=str, default="results/training_outputs"
    )

    # TODO does it make sense to enlarge the img_size to 640?

    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Square resolution used for training/inference. "
        "Must be divisible by 32 (U-Net/ResNet34 constraint). "
        "512 chosen over 384 because source images are ~3000x3264 "
        "and Door Handle is a thin, detail-sensitive class that "
        "loses too much definition under more aggressive downsampling.",
    )
    parser.add_argument(
        "--pad_to_square",
        action="store_true",
        help="Letterbox resize (preserve aspect ratio, pad with background) "
        "instead of a plain stretch-to-square Resize. Optional here since "
        "the raw images are already close to square (~1.09 ratio).",
    )
    parser.add_argument(
        "--no_bbox_crop",
        action="store_true",
        help="Disable cropping to the detected car bounding box before "
        "resizing. Cropping is ON by default: source images have a lot "
        "of empty background canvas, and removing it before resizing "
        "concentrates resolution on the car itself.",
    )
    parser.add_argument(
        "--bbox_threshold",
        type=float,
        default=20.0,
        help="Per-pixel color-distance threshold (summed abs diff over RGB) "
        "used to separate foreground (car) from background canvas.",
    )
    parser.add_argument(
        "--bbox_margin",
        type=float,
        default=0.03,
        help="Extra margin added around the detected bounding box, as a "
        "fraction of box size, to avoid clipping part edges.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
        help="Epochs to train with encoder frozen before fine-tuning it.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base LR for decoder/head params.",
    )
    parser.add_argument(
        "--encoder_lr_factor",
        type=float,
        default=0.1,
        help="Encoder LR = lr * this factor, once unfrozen.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early stopping patience (epochs without val mIoU improvement).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="DataLoader worker processes. Default of 12 assumes a modern "
        "multi-core CPU (e.g. i7-14700K, 28 threads); lower this if you "
        "see CPU contention with other processes. Per-epoch timing is "
        "always printed in the training log, so you can check whether "
        "increasing this further still helps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_fp16",
        action="store_true",
        help="Save the final best checkpoint in fp16 to reduce file size.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a previous best_model.pth to initialize weights from "
        "(e.g. to continue training with more epochs or a fresh LR "
        "schedule -- a 'warm restart'). Automatically skips the "
        "encoder-freeze warm-start phase, since the encoder is already "
        "fine-tuned. This does NOT restore optimizer/scheduler state or "
        "epoch count -- it starts a fresh optimizer and LR schedule "
        "initialized from these weights, which is the standard and "
        "usually preferable way to continue: it lets the model escape "
        "the exact minimum it settled into and potentially find a better "
        "one, rather than just resuming an already-annealed-to-near-zero LR.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = Path(args.output_dir) / (f"training_run_{datetime_stamp}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard: local-only metrics dashboard. No data leaves the machine —
    # important given the dataset's confidentiality requirement. View live at
    # http://localhost:6006 while training runs, via: tensorboard --logdir runs
    writer = SummaryWriter(log_dir=output_dir)

    # --- Data split ---
    train_images, train_masks, val_images, val_masks = split_dataset(
        Path(args.images_dir),
        Path(args.masks_dir),
        args.val_fraction,
        args.seed,
    )
    print(f"Train: {len(train_images)} images | Val: {len(val_images)} images")

    save_run_config(
        args,
        device,
        output_dir,
        datetime_stamp,
        num_train=len(train_images),
        num_val=len(val_images),
    )

    crop_to_bbox = not args.no_bbox_crop
    train_ds = CarPartsDataset(
        train_images,
        train_masks,
        get_transforms(True, args.img_size, args.pad_to_square),
        crop_to_bbox=crop_to_bbox,
        bbox_threshold=args.bbox_threshold,
        bbox_margin=args.bbox_margin,
    )
    val_ds = CarPartsDataset(
        val_images,
        val_masks,
        get_transforms(False, args.img_size, args.pad_to_square),
        crop_to_bbox=crop_to_bbox,
        bbox_threshold=args.bbox_threshold,
        bbox_margin=args.bbox_margin,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Class weights (train split only, avoids val leakage) ---
    class_weights = compute_class_weights(train_masks).to(device)
    criterion = DiceCELoss(class_weights)

    # TODO what about using resnet50 as encoder?
    # It has more parameters and might improve performance, but it will also
    # increase training time and memory usage. We can experiment with both
    # resnet34 and resnet50 to see which one gives better results for our specific dataset.

    # --- Model ---
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",  # overwritten below if --resume_from is set
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    resumed = args.resume_from is not None
    if resumed:
        resume_path = Path(args.resume_from)
        state_dict = torch.load(resume_path, map_location=device)
        # Checkpoints saved with --save_fp16 store half-precision weights; training
        # itself always keeps fp32 master weights (AMP only autocasts internally),
        # so upcast before loading if needed.
        first_dtype = next(iter(state_dict.values())).dtype
        if first_dtype == torch.float16:
            state_dict = {k: v.float() for k, v in state_dict.items()}
            print(
                f"Resuming from {resume_path} (stored as fp16, upcast to fp32 for training)."
            )
        else:
            print(f"Resuming from {resume_path} (fp32 weights).")
        model.load_state_dict(state_dict)

    # Freeze encoder initially: train decoder/head on top of pretrained features.
    # Skipped when resuming -- the encoder is already fine-tuned, so re-freezing
    # it would waste epochs re-deriving what it already learned. --freeze_epochs
    # is ignored in this case (a notice is printed below if it was set non-zero).
    if not resumed:
        for param in model.encoder.parameters():
            param.requires_grad = False
    effective_freeze_epochs = -1 if resumed else args.freeze_epochs
    if resumed and args.freeze_epochs != 0:
        print(
            f"Note: --freeze_epochs={args.freeze_epochs} is ignored because "
            f"--resume_from was set; encoder starts unfrozen."
        )

    def build_optimizer(encoder_frozen: bool):
        if encoder_frozen:
            params = [
                {
                    "params": [
                        p for p in model.parameters() if p.requires_grad
                    ],
                    "lr": args.lr,
                }
            ]
        else:
            params = [
                {
                    "params": model.encoder.parameters(),
                    "lr": args.lr * args.encoder_lr_factor,
                },
                {"params": model.decoder.parameters(), "lr": args.lr},
                {"params": model.segmentation_head.parameters(), "lr": args.lr},
            ]
        return torch.optim.AdamW(params, weight_decay=args.weight_decay)

    # A resumed run starts a FRESH optimizer + LR schedule from full peak LR
    # (a "warm restart") rather than continuing the previous run's annealed,
    # near-zero LR -- this gives the model a real chance to move away from
    # the minimum it settled into, instead of just polishing it further.
    optimizer = build_optimizer(encoder_frozen=not resumed)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_miou = -1.0
    epochs_without_improve = 0
    history = []
    epoch_durations = []

    training_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Unfreeze encoder after the warm-start phase and rebuild optimizer
        # with a lower encoder LR (differential fine-tuning). Never triggers
        # on a resumed run (effective_freeze_epochs == -1), since the encoder
        # already starts unfrozen in that case.
        if epoch == effective_freeze_epochs:
            print(f"Epoch {epoch}: unfreezing encoder for fine-tuning.")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = build_optimizer(encoder_frozen=False)
            scheduler = build_scheduler(
                optimizer, args.epochs - epoch, args.warmup_epochs
            )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, mean_iou, per_class_iou = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = args.epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_duration

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} | lr={current_lr:.2e} "
            f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mIoU={mean_iou:.4f} "
            f"| epoch_time={epoch_duration:.1f}s | avg={avg_epoch_duration:.1f}s/epoch "
            f"| ETA={eta_seconds / 60:.1f}min (if run to --epochs, ignoring early stopping)"
        )
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            print(f"    {name:15s} IoU={iou:.4f}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mIoU": mean_iou,
                "epoch_seconds": epoch_duration,
                "per_class_iou": dict(zip(CLASS_NAMES, per_class_iou.tolist())),
            }
        )

        # --- TensorBoard logging ---
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("IoU/mean_val", mean_iou, epoch + 1)
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            writer.add_scalar(f"IoU_per_class/{name}", iou, epoch + 1)
        writer.add_scalar("LR/current", current_lr, epoch + 1)
        writer.add_scalar("Time/epoch_seconds", epoch_duration, epoch + 1)

        if mean_iou > best_miou:
            best_miou = mean_iou
            epochs_without_improve = 0
            state_dict = model.state_dict()
            if args.save_fp16:
                state_dict = {
                    k: v.half() if v.is_floating_point() else v
                    for k, v in state_dict.items()
                }
            torch.save(state_dict, output_dir / "best_model.pth")
            print(f"  -> New best model saved (val_mIoU={best_miou:.4f})")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)."
                )
                break

    total_training_time = time.time() - training_start
    with open(
        output_dir / f"training_history_{datetime_stamp}.json",
        "w",
    ) as f:
        json.dump(history, f, indent=2)
    writer.close()

    print(
        f"Training complete in {total_training_time / 60:.1f} minutes "
        f"({len(epoch_durations)} epochs run, avg {sum(epoch_durations) / len(epoch_durations):.1f}s/epoch)."
    )
    print(f"Best val mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
