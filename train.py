"""
CoralVisionNet-IO — Local GPU Training Script
==============================================
Tri-stream architecture: ResNet50 + ViT-B/16 + SpectralNet → Gated Fusion → 4-class classifier

Dataset layout expected:
    Merged_Coral_Dataset/
        train/
            Bleached/
            Dead/
            Healthy/
            PartiallyBleached/
        val/   (same sub-folders)
        test/  (same sub-folders)

Run:
    python train.py
    python train.py --data_path ./Merged_Coral_Dataset --epochs 25 --batch_size 16 --phase 1
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")        # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ─── Project imports ─────────────────────────────────────────────────────────
from models.model import CoralVisionNet
from utils.preprocessing import UnderwaterPreprocessor, get_training_augmentation
from utils.loss import FocalLoss
from evaluation.eval import evaluate_and_save


# =============================================================================
# ⚙️  Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoralVisionNet-IO Local GPU Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path",   type=str, default="./Merged_Coral_Dataset",
                   help="Path to dataset root (must contain train/val/test sub-dirs).")
    p.add_argument("--output_dir",  type=str, default="./outputs",
                   help="Directory to save checkpoints, history, and plots.")
    p.add_argument("--epochs",      type=int, default=25)
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Per-GPU batch size. Reduce to 8 if OOM.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers. Set 0 on Windows if you see multiprocessing errors.")
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--phase",       type=int, default=1, choices=[1, 2],
                   help="1=train fusion+head only  2=also fine-tune backbone layers.")
    p.add_argument("--switch_epoch",type=int, default=3,
                   help="Epoch at which Phase-2 fine-tuning kicks in automatically.")
    p.add_argument("--lr",          type=float, default=None,
                   help="Override learning rate (default: 3e-4 phase1 / 8e-5 phase2).")
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--loss_type",   type=str, default="focal", choices=["ce", "focal"],
                   help="ce=CrossEntropy  focal=FocalLoss (better for PartiallyBleached).")
    p.add_argument("--no_eval",     action="store_true",
                   help="Skip final test-set evaluation.")
    return p.parse_args()


# =============================================================================
# 🧰  Utilities
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True   # fastest convolution algorithms


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger("CoralTrain")


# =============================================================================
# 📂  Dataset
# =============================================================================

CLASS_NAMES = ["Bleached", "Dead", "Healthy", "PartiallyBleached"]


class CoralDataset(Dataset):
    """
    Image-folder-style dataset for the Merged_Coral_Dataset.

    Preprocessing pipeline:
        1. Read image with OpenCV (BGR)
        2. Convert to RGB
        3. UnderwaterPreprocessor  → gray-world + CLAHE + resize → RGB tensor + LAB tensor
        4. Optional augmentation   → applied to RGB tensor only

    Returns:
        rgb_tensor  (3, H, W)  ImageNet-normalised
        lab_tensor  (3, H, W)  [0, 1]
        label       int        class index
    """

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir: str, split: str = "train",
                 image_size: int = 224, augment: bool = False):
        self.root = Path(root_dir) / split
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset split '{split}' not found at {self.root}. "
                "Please check your --data_path argument."
            )

        self.preprocessor = UnderwaterPreprocessor(image_size=image_size)
        self.augment = augment
        self.aug_transform = get_training_augmentation() if augment else None

        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        for idx, d in enumerate(class_dirs):
            self.class_to_idx[d.name] = idx
            self.idx_to_class[idx] = d.name
            for img_path in d.iterdir():
                if img_path.suffix.lower() in self.VALID_EXT:
                    self.samples.append((str(img_path), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        if img is None:
            # Corrupted / unreadable image → return a black BGR placeholder
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Pass BGR directly — UnderwaterPreprocessor handles BGR→RGB internally

        rgb, lab = self.preprocessor(img)

        if self.augment and self.aug_transform is not None:
            rgb = self.aug_transform(rgb)

        return rgb, lab, label

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for the loss function."""
        counts = np.zeros(len(self.class_to_idx), dtype=np.float64)
        for _, lbl in self.samples:
            counts[lbl] += 1
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_to_idx)   # normalise
        return torch.FloatTensor(weights)

    def print_summary(self, split: str, logger: logging.Logger) -> None:
        counts = {}
        for _, lbl in self.samples:
            name = self.idx_to_class[lbl]
            counts[name] = counts.get(name, 0) + 1
        logger.info(f"  [{split:5s}] {len(self.samples):>6,} images → {counts}")


# =============================================================================
# 🧠  Model Setup (Phase-Aware Freezing)
# =============================================================================

def build_model(phase: int, device: torch.device,
                logger: logging.Logger) -> CoralVisionNet:
    model = CoralVisionNet(num_classes=4, pretrained=True,
                           dropout=0.3, mode="full")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Phase 1 → train fusion + classifier head + randomly initialized spectral_stream
    if phase == 1:
        for name, param in model.named_parameters():
            if "fusion" in name or "classifier" in name or "spectral_stream" in name:
                param.requires_grad = True
        logger.info("✅ Phase 1 | Training: fusion + classifier + spectral_stream")

    # Phase 2 → fusion + classifier + ResNet layer4 + last ViT block
    elif phase == 2:
        _unfreeze_phase2(model, logger)

    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Total Params : {total:,}")
    logger.info(f"📊 Trainable    : {trainable:,}  ({100 * trainable / total:.2f}%)")
    logger.info(f"🖥️  Device       : {next(model.parameters()).device}")
    return model


def _unfreeze_phase2(model: CoralVisionNet,
                     logger: logging.Logger) -> None:
    """Unfreeze fusion, classifier, ResNet layer4, last ViT encoder block, and SpectralNet."""
    for name, param in model.named_parameters():
        if "fusion" in name or "classifier" in name or "spectral_stream" in name:
            param.requires_grad = True

    if hasattr(model, "spatial_stream") and model.spatial_stream is not None:
        for p in model.spatial_stream.backbone[7].parameters():
            p.requires_grad = True
        logger.info("✅ Unfroze ResNet layer4")

    if hasattr(model, "contextual_stream") and model.contextual_stream is not None:
        vit = model.contextual_stream.backbone
        for p in vit.encoder.layers[-1].parameters():
            p.requires_grad = True
        for p in vit.encoder.ln.parameters():
            p.requires_grad = True
        logger.info("✅ Unfroze ViT last encoder block + LN")

    logger.info("✅ Phase 2 | Fine-tuning enabled")


# =============================================================================
# 🛑  Early Stopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter  = 0
        self.best_f1  = 0.0
        self.should_stop = False

    def __call__(self, f1: float) -> None:
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# =============================================================================
# 🚂  Train / Validate epoch functions
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion,
                scaler, device, logger) -> float:
    model.train()
    
    # [CRITICAL FIX] Keep frozen BatchNorm layers in eval() mode.
    # Otherwise, updating running stats without updating the frozen conv weights 
    # destroys the representation during validation, causing massive overfitting/accuracy drops!
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(module, 'weight') and module.weight is not None:
                if not module.weight.requires_grad:
                    module.eval()

    loss_sum, correct, total = 0.0, 0, 0

    for i, (rgb, lab, labels) in enumerate(loader):
        rgb    = rgb.to(device, non_blocking=True)
        lab    = lab.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=False):
            logits = model(rgb, lab)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        # Gradient clipping — prevents spike instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()   # OneCycleLR steps per batch, not per epoch

        _, pred  = logits.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
        loss_sum += loss.item() * labels.size(0)

        if (i + 1) % 100 == 0:
            logger.info(f"  Batch {i+1:>4}/{len(loader)} | "
                        f"Acc {100. * correct / total:.1f}% | "
                        f"Loss {loss_sum / total:.4f}")

    return 100.0 * correct / total, loss_sum / total


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []

    for rgb, lab, labels in loader:
        rgb    = rgb.to(device, non_blocking=True)
        lab    = lab.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=False):
            logits = model(rgb, lab)
            loss   = criterion(logits, labels)

        _, pred  = logits.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
        loss_sum += loss.item() * labels.size(0)

        preds_all.extend(pred.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = 100.0 * correct / total
    f1  = f1_score(labels_all, preds_all, average="weighted")
    avg_loss = loss_sum / total
    return acc, f1, avg_loss


# =============================================================================
# 📈  Plot Training Curves
# =============================================================================

def plot_history(history: dict, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if history.get("train_loss") and history.get("val_loss"):
        axes[0].plot(history["train_loss"], "o-", label="Train")
        axes[0].plot(history["val_loss"],   "s-", label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], "o-", label="Train")
    axes[1].plot(history["val_acc"],   "s-", label="Val")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_f1"], "s-", color="green", label="Val F1")
    axes[2].set_title("Val F1 (weighted)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"📊 Training curves saved → {out_path}")


# =============================================================================
# 🚀  Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    logger = setup_logging(args.output_dir)

    # ─── GPU / Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 65)
    logger.info("🪸  CoralVisionNet-IO  —  Local GPU Training")
    logger.info("=" * 65)
    logger.info(f"🚀 Device : {device}")
    if device.type == "cuda":
        logger.info(f"🔥 GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("⚠️  CUDA not available — training on CPU will be very slow!")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Dataset verification ─────────────────────────────────────────────────
    data_path = os.path.abspath(args.data_path)
    logger.info(f"\n📂 Dataset path : {data_path}")
    for split in ("train", "val", "test"):
        sp = os.path.join(data_path, split)
        if not os.path.isdir(sp):
            logger.error(f"   ⚠️ MISSING split: {sp}")
            sys.exit(1)

    # ─── Datasets ────────────────────────────────────────────────────────────
    logger.info("\n🔄 Loading datasets …")
    train_ds = CoralDataset(data_path, "train", args.image_size, augment=True)
    val_ds   = CoralDataset(data_path, "val",   args.image_size, augment=False)
    test_ds  = CoralDataset(data_path, "test",  args.image_size, augment=False)

    for ds, split in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
        ds.print_summary(split, logger)

    # ─── DataLoaders ─────────────────────────────────────────────────────────
    # Windows + multiprocessing: if you see "BrokenPipeError" set --num_workers 0
    loader_kw = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        persistent_workers = (args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **loader_kw)

    logger.info(f"\n✅ Train batches : {len(train_loader):,}")
    logger.info(f"✅ Val batches   : {len(val_loader):,}")
    logger.info(f"✅ Test batches  : {len(test_loader):,}")

    # ─── Model ───────────────────────────────────────────────────────────────
    logger.info("\n🧠 Building model …")
    model = build_model(args.phase, device, logger)

    if args.resume:
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)

    # ─── Loss + Optimizer + Scheduler ────────────────────────────────────────
    class_weights = train_ds.get_class_weights().to(device)
    # Re-normalise to be safe
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    logger.info(f"\n📊 Class Weights : {class_weights.tolist()}")

    if args.loss_type == "focal":
        criterion = FocalLoss(
            weight = class_weights,
            gamma  = 2.0,            # focuses on hard/minority examples
        )
        logger.info("🎯 Loss : FocalLoss (gamma=2.0, class-weighted)")
    else:
        criterion = nn.CrossEntropyLoss(
            weight          = class_weights,
            label_smoothing = 0.05,
        )
        logger.info("🎯 Loss : CrossEntropyLoss (label_smoothing=0.05, class-weighted)")

    lr = args.lr if args.lr is not None else (3e-4 if args.phase == 1 else 8e-5)
    logger.info(f"\n⚙️  Learning rate : {lr}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = lr,
        weight_decay = 3e-4,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr            = lr,
        steps_per_epoch   = len(train_loader),
        epochs            = args.epochs,
        pct_start         = 0.15,
        anneal_strategy   = "cos",
        div_factor        = 8,
        final_div_factor  = 200,
    )

    # Mixed-precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ─── Training History ─────────────────────────────────────────────────────
    history = {
        "train_acc"  : [],
        "train_loss" : [],
        "val_acc"    : [],
        "val_loss"   : [],
        "val_f1"     : [],
    }

    best_path    = os.path.join(args.output_dir, "best_model.pth")
    history_path = os.path.join(args.output_dir, "training_history.json")
    best_val_acc = 0.0
    early_stop   = EarlyStopping(patience=5)

    # ─── Training Loop ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("🚀 Training started")
    logger.info("=" * 65)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # ── Automatic Phase-2 switch ──────────────────────────────────────────
        if epoch == args.switch_epoch and args.phase == 1:
            logger.info("\n🔥 AUTO-SWITCHING TO PHASE 2 (fine-tuning) …")
            _unfreeze_phase2(model, logger)

            # Reset optimizer with lower LR for fine-tuning
            fine_lr = 5e-5
            for g in optimizer.param_groups:
                g["lr"] = fine_lr
            logger.info(f"   LR set to {fine_lr}")

        logger.info(f"\n─── Epoch {epoch + 1}/{args.epochs} ───────────────────────")

        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, logger
        )
        val_acc, val_f1, val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start

        # Record history
        history["train_acc"].append(round(train_acc, 4))
        history["train_loss"].append(round(train_loss, 6))
        history["val_acc"].append(round(val_acc, 4))
        history["val_loss"].append(round(val_loss, 6))
        history["val_f1"].append(round(val_f1, 6))

        logger.info(
            f"\nEpoch {epoch + 1}/{args.epochs}  [{elapsed:.0f}s]\n"
            f"   Train → acc {train_acc:.2f}%  loss {train_loss:.4f}\n"
            f"   Val   → acc {val_acc:.2f}%    loss {val_loss:.4f}  F1 {val_f1:.4f}"
        )

        # Save best model (by val accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.info(f"   💾 Best model saved  ({val_acc:.2f}%)")

        # Persist history every epoch in case of crash
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

        # Early stopping on val F1
        early_stop(val_f1)
        if early_stop.should_stop:
            logger.info(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
            break

    # ─── Post-training Plots ──────────────────────────────────────────────────
    logger.info("\n📊 Saving training curves …")
    plot_history(history, args.output_dir)

    logger.info(f"\n✅ Training complete — best val acc: {best_val_acc:.2f}%")
    logger.info(f"   Model   → {best_path}")
    logger.info(f"   History → {history_path}")

    # ─── Final Test-Set Evaluation ────────────────────────────────────────────
    if not args.no_eval:
        logger.info("\n" + "=" * 65)
        logger.info("🔍 Running final evaluation on test set …")
        logger.info("=" * 65)

        # Load the best checkpoint
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )

        class_names = [test_ds.idx_to_class[i]
                       for i in range(len(test_ds.class_to_idx))]

        results = evaluate_and_save(model, test_loader, device,
                                    class_names, args.output_dir)

        logger.info("\n" + "=" * 50)
        logger.info("  📊 FINAL TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"  Accuracy : {results['accuracy']:.2f}%")
        logger.info(f"  F1 Score : {results['f1_score']:.4f}")
        logger.info(f"  Loss     : {results['loss']:.4f}")
        logger.info(f"  ECE      : {results['ece']:.4f}")
        if results.get("gating_weights"):
            for k, v in results["gating_weights"].items():
                logger.info(f"  Gate {k}  : {v:.4f}")

        # Save combined results JSON
        results_path = os.path.join(args.output_dir, "training_results.json")
        combined = {
            "accuracy"        : results["accuracy"],
            "f1_score"        : results["f1_score"],
            "loss"            : results["loss"],
            "ece"             : results.get("ece"),
            "gating_weights"  : results.get("gating_weights"),
            "confusion_matrix": results["confusion_matrix"],
            "epochs_trained"  : len(history["train_acc"]),
            "best_val_acc"    : round(best_val_acc, 2),
            "history"         : history,
        }
        with open(results_path, "w") as fh:
            json.dump(combined, fh, indent=2)
        logger.info(f"\n✅ All results saved to: {args.output_dir}")


# =============================================================================

if __name__ == "__main__":
    # Windows multiprocessing fix
    import multiprocessing
    multiprocessing.freeze_support()
    main()
