"""
models/fusion/fusion.py — Multimodal fusion of video, audio, and text scores.

Three strategies (selectable):
  1. Weighted average      — simple, fast, interpretable
  2. Late fusion MLP       — learns combination weights from data
  3. Attention-based       — attends over modality embeddings

Default: weighted average with domain-tuned weights.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from loguru import logger

from backend.config import settings


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ModalityResult:
    """Structured result from a single modality detector."""
    modality: str                               # "video" | "audio" | "text"
    fake_probability: float
    confidence: float = 1.0                     # how much to trust this result
    explanation: List[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)     # full modality output


@dataclass
class FusionResult:
    """Final fusion output."""
    label: str                                  # "REAL" | "FAKE"
    fake_probability: float
    confidence: str                             # "HIGH" | "MEDIUM" | "LOW"
    modality_scores: dict
    explanation: List[str]
    flag: str                                   # "FAKE" | "REAL" | "UNCERTAIN"


# ─── Weighted Average Fusion ──────────────────────────────────────────────────

class WeightedFusion:
    """
    Simple weighted average.
    Weights can be adaptive: if a modality is unavailable or unreliable
    its weight is redistributed to the remaining modalities.
    """

    def __init__(
        self,
        video_weight: float = settings.VIDEO_WEIGHT,
        audio_weight: float = settings.AUDIO_WEIGHT,
        text_weight:  float = settings.TEXT_WEIGHT,
    ):
        self.weights = {
            "video": video_weight,
            "audio": audio_weight,
            "text":  text_weight,
        }

    def fuse(self, results: List[ModalityResult]) -> float:
        # Log which modalities are being used
        active = [r.modality for r in results]
        logger.info(f"Fusion using: {active}")

        total_w = sum(self.weights[r.modality] * r.confidence for r in results)
        if total_w == 0:
            return 0.5
        score = sum(
            self.weights[r.modality] * r.confidence * r.fake_probability
            for r in results
        ) / total_w
        return float(np.clip(score, 0.0, 1.0))


# ─── Late-Fusion MLP ──────────────────────────────────────────────────────────

class LateFusionMLP(nn.Module):
    """
    Learns to combine modality scores.
    Input:  3-dim vector [video_prob, audio_prob, text_prob]
    Output: scalar fake probability
    """

    def __init__(self, input_dim: int = 3, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(scores)).squeeze(-1)


# ─── Attention Fusion ─────────────────────────────────────────────────────────

class AttentionFusion(nn.Module):
    """
    Modality-level cross-attention.
    Each modality acts as a query over the others.
    Input:  (B, n_modalities, embed_dim)
    Output: scalar fake probability
    """

    def __init__(self, embed_dim: int = 16, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        # Project scalar scores to embed_dim
        self.proj = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: (B, 3)  → (B, 3, 1) → (B, 3, D)
        x = self.proj(scores.unsqueeze(-1))
        x, _ = self.attn(x, x, x)
        return torch.sigmoid(self.head(x)).squeeze(-1)


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class FusionOrchestrator:
    """
    Combines all modality results into a final verdict.
    Adds rule-based overrides for edge cases.
    """

    def __init__(self, strategy: str = "weighted"):
        if strategy == "weighted":
            self._fuser = WeightedFusion()
        elif strategy == "mlp":
            self._mlp = LateFusionMLP()
            self._mlp.eval()
            self._fuser = None
        elif strategy == "attention":
            self._attn = AttentionFusion()
            self._attn.eval()
            self._fuser = None
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        self.strategy = strategy
        logger.info(f"Fusion strategy: {strategy}")

    def combine(self, results: List[ModalityResult]) -> FusionResult:
        """
        Merge modality results → final verdict.
        """
        if not results:
            raise ValueError("No modality results to fuse")

        # ── Compute fused score ───────────────────────────────────────────────
        if self.strategy == "weighted":
            fake_prob = self._fuser.fuse(results)
        elif self.strategy == "mlp":
            scores = torch.tensor(
                [r.fake_probability for r in results], dtype=torch.float
            ).unsqueeze(0)
            with torch.no_grad():
                fake_prob = float(self._mlp(scores)[0])
        elif self.strategy == "attention":
            scores = torch.tensor(
                [r.fake_probability for r in results], dtype=torch.float
            ).unsqueeze(0)
            with torch.no_grad():
                fake_prob = float(self._attn(scores)[0])

        # ── Rule-based override ───────────────────────────────────────────────
        # If any single modality is very highly confident in FAKE → boost
        very_fake = [r for r in results if r.fake_probability > 0.90]
        if very_fake:
            fake_prob = max(fake_prob, 0.85)

        # If all modalities agree on REAL strongly → pull down
        all_real = all(r.fake_probability < 0.2 for r in results)
        if all_real:
            fake_prob = min(fake_prob, 0.15)

        # ── Label & confidence ────────────────────────────────────────────────
        label = "FAKE" if fake_prob >= settings.FAKE_THRESHOLD else "REAL"
        if fake_prob >= settings.HIGH_CONFIDENCE or fake_prob <= (1 - settings.HIGH_CONFIDENCE):
            conf = "HIGH"
        elif 0.40 <= fake_prob <= 0.60:
            conf = "LOW"       # uncertain band
        else:
            conf = "MEDIUM"

        flag = "UNCERTAIN" if conf == "LOW" else label

        # ── Collect explanations ──────────────────────────────────────────────
        explanation = [f"Final score: {fake_prob:.2%} → {label} ({conf} confidence)"]
        for r in results:
            explanation.extend(r.explanation)

        modality_scores = {
            r.modality: {
                "fake_probability": round(r.fake_probability, 4),
                "confidence": r.confidence,
            }
            for r in results
        }

        return FusionResult(
            label=label,
            fake_probability=round(fake_prob, 4),
            confidence=conf,
            modality_scores=modality_scores,
            explanation=explanation,
            flag=flag,
        )