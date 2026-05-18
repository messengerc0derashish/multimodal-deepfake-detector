"""
training/train_video.py — Training pipeline for the video deepfake detector.

Supports FaceForensics++ and Celeb-DF dataset formats.

Usage:
    python -m training.train_video \
        --data_dir /path/to/dataset \
        --epochs 30 \
        --batch_size 32 \
        --backbone efficientnet_b4 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from loguru import logger
import mlflow

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.video.video_model import VideoDeepfakeDetector
from backend.config import settings


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Dataset of pre-extracted face crops.
    Expected structure:
        data_dir/
            real/  *.jpg  (real face crops)
            fake/  *.jpg  (fake face crops)

    For FaceForensics++: use the provided extraction scripts first.
    For Celeb-DF: convert to this flat format.
    """

    TRANSFORM_TRAIN = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    TRANSFORM_VAL = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_dir: Path, split: str = "train", max_samples_per_class: int = None):
        self.split = split
        self.transform = self.TRANSFORM_TRAIN if split == "train" else self.TRANSFORM_VAL
        self.samples: list[Tuple[Path, int]] = []

        for label, cls in [(0, "real"), (1, "fake")]:
            cls_dir = data_dir / cls
            if not cls_dir.exists():
                logger.warning(f"Directory not found: {cls_dir}")
                continue

            cls_samples = []
            for img_path in cls_dir.glob("*.jpg"):
                cls_samples.append((img_path, label))
            for img_path in cls_dir.glob("*.png"):
                cls_samples.append((img_path, label))

            # ── Limit per class if requested ──────────────────────────────────
            if max_samples_per_class is not None and len(cls_samples) > max_samples_per_class:
                import random
                random.shuffle(cls_samples)
                cls_samples = cls_samples[:max_samples_per_class]
                logger.info(f"[{cls}] Capped to {max_samples_per_class} samples")

            self.samples.extend(cls_samples)

        logger.info(f"[{split}] {len(self.samples)} face crops loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        t = self.transform(img)
        # Expand to (T=1, C, H, W) for the temporal model
        return t.unsqueeze(0), torch.tensor(label, dtype=torch.float)


# ─── Training ──────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # ── Data ──────────────────────────────────────────────────────────────
        data_dir = Path(args.data_dir)
        full_ds = FaceDataset(data_dir, split="train",
                              max_samples_per_class=args.max_samples_per_class)

        val_size   = int(0.15 * len(full_ds))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        val_ds.dataset.split = "val"

        self.train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = VideoDeepfakeDetector(
            backbone=args.backbone,
            pretrained=True,
            dropout=0.4,
        ).to(self.device)

        # ── Optimiser ─────────────────────────────────────────────────────────
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=args.epochs, eta_min=1e-6,
        )

        # Class-balanced BCE loss
        n_real = sum(1 for _, l in full_ds.samples if l == 0)
        n_fake = len(full_ds) - n_real
        pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.artifact_criterion = nn.BCEWithLogitsLoss()

        self.scaler = torch.cuda.amp.GradScaler(enabled="cuda" in str(self.device))
        self.best_val_auc = 0.0
        self.patience_counter = 0

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir / "video_model.pth"

    def train_epoch(self) -> dict:
        self.model.train()
        losses, preds, labels = [], [], []

        for frames, label in tqdm(self.train_loader, desc="  Train", leave=False):
            frames = frames.to(self.device)       # (B, T, C, H, W)
            label  = label.to(self.device)        # (B,)

            self.optimiser.zero_grad()
            with torch.cuda.amp.autocast(enabled="cuda" in str(self.device)):
                out = self.model(frames)
                main_loss = self.criterion(out["logit"], label)

                # Auxiliary artifact loss (frame-level)
                art_label = label.unsqueeze(1).expand_as(out["artifact_logits"])
                art_loss  = self.artifact_criterion(out["artifact_logits"], art_label)
                loss = main_loss + 0.3 * art_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()

            losses.append(loss.item())
            preds.extend(out["prob"].detach().cpu().tolist())
            labels.extend(label.cpu().tolist())

        return self._metrics(losses, preds, labels)

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.model.eval()
        losses, preds, labels = [], [], []

        for frames, label in tqdm(self.val_loader, desc="  Val  ", leave=False):
            frames = frames.to(self.device)
            label  = label.to(self.device)
            out = self.model(frames)
            loss = self.criterion(out["logit"], label)
            losses.append(loss.item())
            preds.extend(out["prob"].cpu().tolist())
            labels.extend(label.cpu().tolist())

        return self._metrics(losses, preds, labels)

    def _metrics(self, losses, preds, labels) -> dict:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )
        bin_preds = [1 if p > 0.5 else 0 for p in preds]
        return {
            "loss":      np.mean(losses),
            "accuracy":  accuracy_score(labels, bin_preds),
            "precision": precision_score(labels, bin_preds, zero_division=0),
            "recall":    recall_score(labels, bin_preds, zero_division=0),
            "f1":        f1_score(labels, bin_preds, zero_division=0),
            "auc":       roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5,
        }

    def run(self):
        logger.info(f"Training for {self.args.epochs} epochs")
        mlflow.set_experiment("video_deepfake_detection")

        with mlflow.start_run():
            mlflow.log_params(vars(self.args))

            for epoch in range(1, self.args.epochs + 1):
                logger.info(f"Epoch {epoch}/{self.args.epochs}")

                train_m = self.train_epoch()
                val_m   = self.val_epoch()
                self.scheduler.step()

                logger.info(
                    f"  Train — loss={train_m['loss']:.4f} acc={train_m['accuracy']:.3f} "
                    f"f1={train_m['f1']:.3f} auc={train_m['auc']:.3f}"
                )
                logger.info(
                    f"  Val   — loss={val_m['loss']:.4f} acc={val_m['accuracy']:.3f} "
                    f"f1={val_m['f1']:.3f} auc={val_m['auc']:.3f}"
                )

                mlflow.log_metrics(
                    {f"train_{k}": v for k, v in train_m.items()} |
                    {f"val_{k}":   v for k, v in val_m.items()},
                    step=epoch,
                )

                # Save best model
                if val_m["auc"] > self.best_val_auc:
                    self.best_val_auc = val_m["auc"]
                    torch.save(self.model.state_dict(), self.save_path)
                    logger.success(f"  ✓ New best AUC={val_m['auc']:.4f} — saved to {self.save_path}")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.args.patience:
                        logger.warning(f"Early stopping at epoch {epoch}")
                        break

            logger.success(f"Training complete — best val AUC: {self.best_val_auc:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train video deepfake detector")
    p.add_argument("--data_dir",    required=True, help="Path to face crops dataset")
    p.add_argument("--save_dir",    default="data/models", help="Where to save weights")
    p.add_argument("--backbone",    default="efficientnet_b4")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max_samples_per_class", type=int, default=None,
                   help="Limit files per class (real/fake) for faster runs. e.g. 500")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Trainer(args).run()



