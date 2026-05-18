"""
training/train_audio.py — Training pipeline for the audio deepfake detector.

Supports ASVspoof 2019/2021 and FakeAVCeleb dataset formats.

Expected data structure:
    data_dir/
        real/  *.wav  (genuine speech)
        fake/  *.wav  (synthetic / converted speech)

Usage:
    python -m training.train_audio \
        --data_dir /path/to/audio_dataset \
        --epochs 40 \
        --batch_size 64 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from loguru import logger
import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.audio.audio_model import AudioDeepfakeDetector
from backend.utils.extraction import compute_mfcc, compute_mel_spectrogram
from backend.config import settings

import librosa


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """
    Loads WAV files and returns (mfcc_tensor, mel_tensor, label).
    """

    def __init__(self, data_dir: Path, sr: int = 16000, augment: bool = False):
        self.sr = sr
        self.augment = augment
        self.samples: list[Tuple[Path, int]] = []

        for label, cls in [(0, "real"), (1, "fake")]:
            cls_dir = data_dir / cls
            if not cls_dir.exists():
                logger.warning(f"Missing: {cls_dir}")
                continue
            for ext in ["*.wav", "*.flac", "*.mp3"]:
                for p in cls_dir.glob(ext):
                    self.samples.append((p, label))

        logger.info(f"AudioDataset: {len(self.samples)} clips")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        waveform, _ = librosa.load(str(path), sr=self.sr, mono=True,
                                   duration=settings.AUDIO_MAX_DURATION)

        if self.augment:
            waveform = self._augment(waveform)

        mfcc = compute_mfcc(waveform, self.sr)         # (40, 256)
        mel  = compute_mel_spectrogram(waveform, self.sr)  # (128, 256)

        mfcc_t = torch.tensor(mfcc).unsqueeze(0)       # (1, 40, 256)
        mel_t  = torch.tensor(mel).unsqueeze(0)        # (1, 128, 256)
        lbl    = torch.tensor(label, dtype=torch.float)
        return mfcc_t, mel_t, lbl

    def _augment(self, waveform: np.ndarray) -> np.ndarray:
        """Time-domain augmentations."""
        # Additive Gaussian noise
        if np.random.rand() < 0.5:
            waveform = waveform + np.random.randn(*waveform.shape) * 0.003

        # Time stretching
        if np.random.rand() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            waveform = librosa.effects.time_stretch(waveform, rate=rate)

        # Pitch shifting
        if np.random.rand() < 0.3:
            steps = np.random.uniform(-1.5, 1.5)
            waveform = librosa.effects.pitch_shift(waveform, sr=self.sr, n_steps=steps)

        # Volume scaling
        waveform = waveform * np.random.uniform(0.8, 1.2)
        return waveform.astype(np.float32)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class AudioTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Data
        data_dir = Path(args.data_dir)
        full_ds  = AudioDataset(data_dir, augment=True)
        val_size = int(0.15 * len(full_ds))
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
        val_ds.dataset.augment = False

        self.train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        # Model
        self.model = AudioDeepfakeDetector(
            n_mfcc=40, n_mels=128,
            cnn_dim=256, lstm_hidden=256,
            lstm_layers=2, dropout=0.3,
        ).to(self.device)

        # Optimiser + scheduler
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        steps_per_epoch = len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser,
            max_lr=args.lr * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled="cuda" in str(self.device))
        self.best_auc = 0.0
        self.patience_counter = 0

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir / "audio_model.pth"

    def train_epoch(self) -> dict:
        self.model.train()
        losses, preds, labels = [], [], []

        for mfcc, mel, label in tqdm(self.train_loader, desc="  Train", leave=False):
            mfcc  = mfcc.to(self.device)
            mel   = mel.to(self.device)
            label = label.to(self.device)

            self.optimiser.zero_grad()
            with torch.cuda.amp.autocast(enabled="cuda" in str(self.device)):
                out  = self.model(mfcc, mel)
                loss = self.criterion(out["logit"], label)
                # Auxiliary segment loss
                seg_label = label.unsqueeze(1).expand_as(out["segment_logits"])
                seg_loss  = self.criterion(out["segment_logits"], seg_label)
                total = loss + 0.2 * seg_loss

            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.scheduler.step()

            losses.append(total.item())
            preds.extend(out["prob"].detach().cpu().tolist())
            labels.extend(label.cpu().tolist())

        return self._metrics(losses, preds, labels)

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.model.eval()
        losses, preds, labels = [], [], []
        for mfcc, mel, label in tqdm(self.val_loader, desc="  Val  ", leave=False):
            mfcc, mel, label = mfcc.to(self.device), mel.to(self.device), label.to(self.device)
            out  = self.model(mfcc, mel)
            loss = self.criterion(out["logit"], label)
            losses.append(loss.item())
            preds.extend(out["prob"].cpu().tolist())
            labels.extend(label.cpu().tolist())
        return self._metrics(losses, preds, labels)

    def _metrics(self, losses, preds, labels) -> dict:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        bin_preds = [1 if p > 0.5 else 0 for p in preds]
        return {
            "loss": np.mean(losses),
            "accuracy": accuracy_score(labels, bin_preds),
            "f1": f1_score(labels, bin_preds, zero_division=0),
            "auc": roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5,
        }

    def run(self):
        mlflow.set_experiment("audio_deepfake_detection")
        with mlflow.start_run():
            mlflow.log_params(vars(self.args))
            for epoch in range(1, self.args.epochs + 1):
                logger.info(f"Epoch {epoch}/{self.args.epochs}")
                train_m = self.train_epoch()
                val_m   = self.val_epoch()
                logger.info(f"  Train — {train_m}")
                logger.info(f"  Val   — {val_m}")
                mlflow.log_metrics(
                    {f"train_{k}": v for k, v in train_m.items()} |
                    {f"val_{k}": v for k, v in val_m.items()},
                    step=epoch,
                )
                if val_m["auc"] > self.best_auc:
                    self.best_auc = val_m["auc"]
                    torch.save(self.model.state_dict(), self.save_path)
                    logger.success(f"  ✓ Saved — AUC={val_m['auc']:.4f}")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.args.patience:
                        logger.warning("Early stopping")
                        break
        logger.success(f"Best audio AUC: {self.best_auc:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--save_dir",     default="data/models")
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--patience",     type=int,   default=7)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    AudioTrainer(parse_args()).run()