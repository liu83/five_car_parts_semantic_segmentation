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
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_raw = np.array(Image.open(self.mask_paths[idx]))

        # Map raw label values (0,32,64,...) -> class indices (0..5)
        mask = np.zeros_like(mask_raw, dtype=np.uint8)
        for value, index in VALUE_TO_INDEX.items():
            mask[mask_raw == value] = index

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].long()
        return image_t, mask_t


def get_transforms(train: bool, img_size: int):
    mean = (0.485, 0.456, 0.406)  # ImageNet stats, matches pretrained encoder
    std = (0.229, 0.224, 0.225)

    if train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
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
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def split_dataset(
    images_dir: Path, masks_dir: Path, val_fraction: float, seed: int
):
    # TODO maybe consider class distribution when splitting,
    # but for now just random shuffle

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
def main():
    parser = argparse.ArgumentParser(
        description="Train car part segmentation model"
    )
    parser.add_argument("--images_dir", type=str, default="data/images")
    parser.add_argument("--masks_dir", type=str, default="data/masks")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Square resolution used for training/inference.",
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_fp16",
        action="store_true",
        help="Save the final best checkpoint in fp16 to reduce file size.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data split ---
    train_images, train_masks, val_images, val_masks = split_dataset(
        Path(args.images_dir),
        Path(args.masks_dir),
        args.val_fraction,
        args.seed,
    )
    print(f"Train: {len(train_images)} images | Val: {len(val_images)} images")

    train_ds = CarPartsDataset(
        train_images, train_masks, get_transforms(True, args.img_size)
    )
    val_ds = CarPartsDataset(
        val_images, val_masks, get_transforms(False, args.img_size)
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

    # --- Model ---
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    # Freeze encoder initially: train decoder/head on top of pretrained features.
    for param in model.encoder.parameters():
        param.requires_grad = False

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

    optimizer = build_optimizer(encoder_frozen=True)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_miou = -1.0
    epochs_without_improve = 0
    history = []

    for epoch in range(args.epochs):
        # Unfreeze encoder after the warm-start phase and rebuild optimizer
        # with a lower encoder LR (differential fine-tuning).
        if epoch == args.freeze_epochs:
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

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} | lr={current_lr:.2e} "
            f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mIoU={mean_iou:.4f}"
        )
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            print(f"    {name:15s} IoU={iou:.4f}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mIoU": mean_iou,
                "per_class_iou": dict(zip(CLASS_NAMES, per_class_iou.tolist())),
            }
        )

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

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
