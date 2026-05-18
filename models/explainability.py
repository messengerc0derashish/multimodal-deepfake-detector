"""
models/explainability.py — Grad-CAM heatmaps and audio segment highlighting.

Produces visual explanations for why a video was flagged as deepfake:
  - Grad-CAM heatmap overlaid on face crops (video)
  - Suspicious audio segment markers (audio)
  - Confidence bar chart across modalities
"""

import io
import base64
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from loguru import logger

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not installed — heatmaps disabled")


# ─── Grad-CAM ────────────────────────────────────────────────────────────────

def generate_gradcam(
    model: torch.nn.Module,
    face_tensor: torch.Tensor,          # (1, C, H, W) normalised
    target_layer: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for a face tensor.
    Returns uint8 heatmap RGB image (H, W, 3) or None.
    """
    if not GRADCAM_AVAILABLE:
        return None

    try:
        if target_layer is None:
            # Auto-pick last conv block for EfficientNet / ViT
            target_layer = _get_target_layer(model)

        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=face_tensor.to(device))    # (1, H, W)
        grayscale_cam = grayscale_cam[0]                             # (H, W)

        # Unnormalise face for overlay
        rgb = _unnormalize(face_tensor[0].cpu().numpy())             # (H, W, 3)
        heatmap = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        return heatmap
    except Exception as e:
        logger.debug(f"Grad-CAM failed: {e}")
        return None


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Auto-detect the last conv layer in a model."""
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layers found")
    return last_conv


def _unnormalize(
    tensor: np.ndarray,
    mean: Tuple = (0.485, 0.456, 0.406),
    std:  Tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverse ImageNet normalisation → [0,1] float RGB."""
    out = tensor.transpose(1, 2, 0)     # C,H,W → H,W,C
    out = out * np.array(std) + np.array(mean)
    return np.clip(out, 0, 1).astype(np.float32)


# ─── Audio Highlighting ───────────────────────────────────────────────────────

def plot_audio_analysis(
    waveform: np.ndarray,
    sr: int,
    suspicious_segments: List[dict],
    fake_probability: float,
    save_path: Optional[Path] = None,
) -> str:
    """
    Plot waveform + mel spectrogram with suspicious segments highlighted.
    Returns base64-encoded PNG string.
    """
    import librosa

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), facecolor="#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    duration = len(waveform) / sr
    t = np.linspace(0, duration, len(waveform))

    # ── Waveform ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(t, waveform, color="#4fc3f7", linewidth=0.4, alpha=0.85)
    ax1.set_ylabel("Amplitude", color="white", fontsize=9)
    ax1.set_xlim(0, duration)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.set_title(
        f"Audio Analysis — {'FAKE' if fake_probability > 0.5 else 'REAL'} "
        f"({fake_probability:.1%})",
        color="white", fontsize=11,
    )

    # Highlight suspicious
    for seg in suspicious_segments:
        ax1.axvspan(seg["start_s"], seg["end_s"], alpha=0.35, color="#ef5350")

    # ── Mel Spectrogram ───────────────────────────────────────────────────────
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    ax2 = axes[1]
    img = ax2.imshow(
        log_mel, aspect="auto", origin="lower",
        extent=[0, duration, 0, sr // 2 / 1000],
        cmap="magma", vmin=-80, vmax=0,
    )
    ax2.set_ylabel("Freq (kHz)", color="white", fontsize=9)
    ax2.set_xlabel("Time (s)", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)

    # Highlight suspicious on spectrogram
    for seg in suspicious_segments:
        ax2.axvspan(seg["start_s"], seg["end_s"], alpha=0.3, color="#ef5350",
                    ymin=0, ymax=1)

    plt.colorbar(img, ax=ax2, label="dB").ax.yaxis.label.set_color("white")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")

    if save_path:
        save_path.write_bytes(base64.b64decode(encoded))

    return encoded


# ─── Modality Confidence Chart ────────────────────────────────────────────────

def plot_modality_breakdown(
    modality_scores: dict,
    final_prob: float,
) -> str:
    """
    Bar chart showing per-modality fake probability.
    Returns base64 PNG.
    """
    labels = list(modality_scores.keys())
    scores = [modality_scores[m]["fake_probability"] for m in labels]
    labels.append("FINAL")
    scores.append(final_prob)

    colors = [
        "#ef5350" if s > 0.5 else "#66bb6a"
        for s in scores
    ]
    colors[-1] = "#ef9a9a" if final_prob > 0.5 else "#a5d6a7"

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(labels, scores, color=colors, edgecolor="none", height=0.5)

    ax.axvline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fake Probability", color="white", fontsize=9)
    ax.set_title("Modality Breakdown", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=9)

    for bar, score in zip(bars, scores):
        ax.text(
            min(score + 0.03, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}", va="center", ha="left", color="white", fontsize=8,
        )

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")