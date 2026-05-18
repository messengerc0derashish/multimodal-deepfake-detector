"""
models/audio/audio_model.py — Audio deepfake detector.

Architecture:
  1. MFCC + log-mel spectrogram feature extraction (Librosa)
  2. 2-branch CNN: one for MFCC, one for mel spectrogram
  3. BiLSTM temporal encoder
  4. Attention pooling
  5. Binary classifier head (real / fake voice)

Detects:
  - Synthetic voice patterns (TTS / voice conversion)
  - Unnatural frequency artifacts
  - Temporal inconsistency in prosody
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, List
from loguru import logger

from backend.config import settings
from backend.utils.extraction import compute_mfcc, compute_mel_spectrogram


# ─── CNN Feature Extractor ────────────────────────────────────────────────────

class SpectralCNN(nn.Module):
    """
    Lightweight 2-D CNN for spectrogram / MFCC feature maps.
    Input:  (B, 1, freq_bins, time_steps)
    Output: (B, D, time_steps')
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2, 1)),       # halve freq axis, keep time

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((2, 1)),

            # Block 3
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, None)),   # collapse freq dim → (B, D, 1, T)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)           # (B, D, 1, T)
        return out.squeeze(2)       # (B, D, T)


# ─── Attention Pooling ────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Soft-attention over time steps → fixed-size vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = torch.softmax(self.query(x), dim=1)   # (B, T, 1)
        return (weights * x).sum(dim=1)                  # (B, D)


# ─── Main Audio Model ─────────────────────────────────────────────────────────

class AudioDeepfakeDetector(nn.Module):
    """
    Dual-branch model: MFCC branch + mel-spectrogram branch → BiLSTM → classifier.

    Input shape per branch:
        mfcc_input: (B, 1, n_mfcc, time)
        mel_input:  (B, 1, n_mels, time)

    Output: scalar logit (sigmoid → fake probability)
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        n_mels: int = 128,
        cnn_dim: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Two CNN branches ──────────────────────────────────────────────────
        self.mfcc_cnn = SpectralCNN(in_channels=1, out_dim=cnn_dim)
        self.mel_cnn  = SpectralCNN(in_channels=1, out_dim=cnn_dim)

        fused_dim = cnn_dim * 2     # concat both branches

        # ── Temporal encoder ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        bilstm_dim = lstm_hidden * 2

        # ── Attention pooling ─────────────────────────────────────────────────
        self.pool = AttentionPool(bilstm_dim)

        # ── Classifier ────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(bilstm_dim),
            nn.Linear(bilstm_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

        # Auxiliary: frame-level fake detector for explainability
        self.segment_head = nn.Linear(bilstm_dim, 1)

    def forward(
        self,
        mfcc: torch.Tensor,     # (B, 1, n_mfcc, T)
        mel: torch.Tensor,      # (B, 1, n_mels, T)
    ) -> dict:
        # CNN feature extraction
        mfcc_feat = self.mfcc_cnn(mfcc)    # (B, D, T)
        mel_feat  = self.mel_cnn(mel)       # (B, D, T)

        # Align time axis (take min length)
        T = min(mfcc_feat.shape[-1], mel_feat.shape[-1])
        mfcc_feat = mfcc_feat[:, :, :T]
        mel_feat  = mel_feat[:, :, :T]

        # Concat along feature dim, then transpose for LSTM
        fused = torch.cat([mfcc_feat, mel_feat], dim=1)   # (B, 2D, T)
        fused = fused.permute(0, 2, 1)                    # (B, T, 2D)

        # BiLSTM
        lstm_out, _ = self.lstm(fused)      # (B, T, 2*hidden)

        # Segment-level scores for explanation
        segment_logits = self.segment_head(lstm_out).squeeze(-1)  # (B, T)

        # Aggregate
        pooled = self.pool(lstm_out)        # (B, 2*hidden)

        # Classification
        logit = self.head(pooled).squeeze(-1)   # (B,)

        return {
            "logit": logit,
            "prob":  torch.sigmoid(logit),
            "segment_logits": segment_logits,
            "segment_probs":  torch.sigmoid(segment_logits),
        }


# ─── Inference Helper ─────────────────────────────────────────────────────────

class AudioInference:
    """High-level wrapper for audio deepfake prediction."""

    def __init__(self, model_path: Optional[Path], device: str = "cpu"):
        self.device = device
        self.model = AudioDeepfakeDetector()
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            logger.info(f"Audio model loaded from {model_path}")
        else:
            logger.warning("No trained audio weights — using random init (for demo)")
        self.model.to(device).eval()

    def _prepare_inputs(
        self, waveform: np.ndarray, sr: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MFCC + mel and return as tensors."""
        mfcc = compute_mfcc(waveform, sr)          # (40, 256)
        mel  = compute_mel_spectrogram(waveform, sr)  # (128, 256)

        mfcc_t = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)   # (1,1,40,256)
        mel_t  = torch.tensor(mel).unsqueeze(0).unsqueeze(0)    # (1,1,128,256)
        return mfcc_t.to(self.device), mel_t.to(self.device)

    @torch.no_grad()
    def predict(self, waveform: np.ndarray, sr: int) -> dict:
        """Predict deepfake probability from a waveform."""
        if len(waveform) < sr * 0.5:
            return {
                "fake_probability": 0.5,
                "explanation": ["Audio too short to analyse"],
                "suspicious_segments": [],
            }

        mfcc_t, mel_t = self._prepare_inputs(waveform, sr)
        out = self.model(mfcc_t, mel_t)

        fake_prob = float(out["prob"][0].cpu())
        seg_probs = out["segment_probs"][0].cpu().tolist()

        explanation = self._build_explanation(fake_prob, seg_probs, sr)
        suspicious  = self._find_suspicious_segments(seg_probs, sr)

        return {
            "fake_probability": round(fake_prob, 4),
            "segment_scores":   [round(s, 4) for s in seg_probs[:20]],
            "suspicious_segments": suspicious,
            "explanation": explanation,
        }

    def _build_explanation(
        self, fake_prob: float, seg_probs: List[float], sr: int
    ) -> List[str]:
        reasons = []
        if fake_prob > 0.8:
            reasons.append(f"Audio shows strong synthetic voice pattern (score={fake_prob:.2f})")
        elif fake_prob > 0.5:
            reasons.append(f"Audio contains possible voice synthesis artifacts (score={fake_prob:.2f})")
        else:
            reasons.append(f"Audio appears to be genuine voice (score={fake_prob:.2f})")

        n_suspicious = sum(1 for s in seg_probs if s > 0.6)
        if n_suspicious > len(seg_probs) * 0.3:
            reasons.append(f"{n_suspicious} of {len(seg_probs)} segments show artificial patterns")

        variance = float(np.var(seg_probs)) if seg_probs else 0.0
        if variance > 0.05:
            reasons.append("Unnatural prosodic variance detected in voice segments")
        return reasons

    def _find_suspicious_segments(
        self, seg_probs: List[float], sr: int, hop_length: int = 512
    ) -> List[dict]:
        """Return time-stamped suspicious segments."""
        suspicious = []
        seg_duration = hop_length / sr   # seconds per LSTM step
        for i, prob in enumerate(seg_probs):
            if prob > 0.6:
                suspicious.append({
                    "start_s": round(i * seg_duration, 2),
                    "end_s":   round((i + 1) * seg_duration, 2),
                    "score":   round(prob, 4),
                })
        return suspicious[:10]   # cap output