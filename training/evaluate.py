"""
training/evaluate.py — Comprehensive evaluation of trained models.

Computes:
  - Accuracy, Precision, Recall, F1, AUC-ROC
  - Confusion matrix
  - Per-threshold analysis
  - Grad-CAM visualisations on worst-case samples

Usage:
    python -m training.evaluate \
        --data_dir /path/to/test_data \
        --video_model data/models/video_model.pth \
        --audio_model data/models/audio_model.pth \
        --output_dir eval_results/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.video.video_model import VideoInference
from models.audio.audio_model import AudioInference
from models.fusion.fusion import FusionOrchestrator, ModalityResult
from backend.config import settings


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_pipeline(
    test_videos: list,      # [(video_path, label_int), ...]
    video_model: VideoInference,
    audio_model: AudioInference,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run full pipeline on all test videos and compute metrics."""
    from backend.utils.extraction import extract_frames, extract_audio
    from models.fusion.fusion import FusionOrchestrator, ModalityResult

    fusion = FusionOrchestrator("weighted")
    y_true, y_score = [], []
    errors = 0

    for video_path, true_label in tqdm(test_videos, desc="Evaluating"):
        try:
            frames   = extract_frames(video_path)
            waveform, sr = extract_audio(video_path)

            v_out = video_model.predict(frames)
            a_out = audio_model.predict(waveform, sr)

            v_res = ModalityResult("video", v_out["fake_probability"],
                                   confidence=1.0 if v_out.get("face_detected") else 0.5)
            a_res = ModalityResult("audio", a_out["fake_probability"], confidence=1.0)

            result = fusion.combine([v_res, a_res])
            y_true.append(true_label)
            y_score.append(result.fake_probability)
        except Exception as e:
            logger.warning(f"Skip {video_path}: {e}")
            errors += 1

    if not y_true:
        logger.error("No predictions — check dataset paths")
        return {}

    y_pred = [1 if s >= 0.5 else 0 for s in y_score]

    metrics = {
        "n_samples":  len(y_true),
        "n_errors":   errors,
        "accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":    round(roc_auc_score(y_true, y_score), 4) if len(set(y_true)) > 1 else None,
        "avg_precision": round(average_precision_score(y_true, y_score), 4)
                         if len(set(y_true)) > 1 else None,
    }

    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    _plot_confusion_matrix(y_true, y_pred, output_dir)
    _plot_roc_curve(y_true, y_score, output_dir)
    _plot_pr_curve(y_true, y_score, output_dir)
    _plot_score_distribution(y_true, y_score, output_dir)
    _plot_threshold_analysis(y_true, y_score, output_dir)

    return metrics

# ─── Video Evaluation ───────────────────────────────────────────────────────────────

def evaluate_video_model(
    test_videos: list,   # [(video_path, label_int), ...]
    video_model,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    """Evaluate only the video model."""

    from backend.utils.extraction import extract_frames

    y_true, y_score = [], []
    errors = 0

    for video_path, true_label in tqdm(test_videos, desc="Evaluating Video"):
        try:
            frames = extract_frames(video_path)

            v_out = video_model.predict(frames)
            score = v_out["fake_probability"]

            y_true.append(true_label)
            y_score.append(score)

        except Exception as e:
            logger.warning(f"Skip {video_path}: {e}")
            errors += 1

    if not y_true:
        logger.error("No predictions — check dataset paths")
        return {}

    y_pred = [1 if s >= 0.5 else 0 for s in y_score]

    metrics = {
        "n_samples": len(y_true),
        "n_errors": errors,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_score), 4) if len(set(y_true)) > 1 else None,
        "avg_precision": round(average_precision_score(y_true, y_score), 4)
                         if len(set(y_true)) > 1 else None,
    }

    logger.info(f"[VIDEO] Metrics: {json.dumps(metrics, indent=2)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "video_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    _plot_confusion_matrix(y_true, y_pred, output_dir, prefix="video")
    _plot_roc_curve(y_true, y_score, output_dir, prefix="video")
    _plot_pr_curve(y_true, y_score, output_dir, prefix="video")
    _plot_score_distribution(y_true, y_score, output_dir, prefix="video")
    _plot_threshold_analysis(y_true, y_score, output_dir, prefix="video")

    return metrics

# ─── Audio Evaluation ───────────────────────────────────────────────────────────────

def evaluate_audio_model(
    test_videos: list,   # [(video_path, label_int), ...]
    audio_model,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    """Evaluate only the audio model."""

    from backend.utils.extraction import extract_audio

    y_true, y_score = [], []
    errors = 0

    for video_path, true_label in tqdm(test_videos, desc="Evaluating Audio"):
        try:
            waveform, sr = extract_audio(video_path)

            a_out = audio_model.predict(waveform, sr)
            score = a_out["fake_probability"]

            y_true.append(true_label)
            y_score.append(score)

        except Exception as e:
            logger.warning(f"Skip {video_path}: {e}")
            errors += 1

    if not y_true:
        logger.error("No predictions — check dataset paths")
        return {}

    y_pred = [1 if s >= 0.5 else 0 for s in y_score]

    metrics = {
        "n_samples": len(y_true),
        "n_errors": errors,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_score), 4) if len(set(y_true)) > 1 else None,
        "avg_precision": round(average_precision_score(y_true, y_score), 4)
                         if len(set(y_true)) > 1 else None,
    }

    logger.info(f"[AUDIO] Metrics: {json.dumps(metrics, indent=2)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "audio_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    _plot_confusion_matrix(y_true, y_pred, output_dir, prefix="audio")
    _plot_roc_curve(y_true, y_score, output_dir, prefix="audio")
    _plot_pr_curve(y_true, y_score, output_dir, prefix="audio")
    _plot_score_distribution(y_true, y_score, output_dir, prefix="audio")
    _plot_threshold_analysis(y_true, y_score, output_dir, prefix="audio")

    return metrics

# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, out_dir: Path, prefix=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"], color="white")
    ax.set_yticklabels(["Real", "Fake"], color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_title("Confusion Matrix", color="white")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] > cm.max() / 2 else "white", fontsize=18)

    filename = f"{prefix}_confusion_matrix.png" if prefix else "confusion_matrix.png"

    plt.colorbar(im, ax=ax).ax.yaxis.label.set_color("white")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, facecolor="#0d0d1a")
    plt.close()
    logger.info("Saved confusion_matrix.png")


def _plot_roc_curve(y_true, y_score, out_dir: Path, prefix=""):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d0d1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot(fpr, tpr, color="#7c6ff7", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1], [0,1], color="#555", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curve", color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white")

    filename = f"{prefix}_roc_curve.png" if prefix else "roc_curve.png"

    
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, facecolor="#0d0d1a")
    plt.close()
    logger.info("Saved roc_curve.png")


def _plot_pr_curve(y_true, y_score, out_dir: Path, prefix = ""):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d0d1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot(rec, prec, color="#66bb6a", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall", color="white")
    ax.set_ylabel("Precision", color="white")
    ax.set_title("Precision-Recall Curve", color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white")

    filename = f"{prefix}_pr_curve.png" if prefix else "pr_curve.png"

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, facecolor="#0d0d1a")
    plt.close()


def _plot_score_distribution(y_true, y_score, out_dir: Path, prefix=""):
    real_scores = [s for s, l in zip(y_score, y_true) if l == 0]
    fake_scores = [s for s, l in zip(y_score, y_true) if l == 1]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d0d1a")
    ax.set_facecolor("#1a1a2e")
    ax.hist(real_scores, bins=40, alpha=0.7, color="#66bb6a", label="Real")
    ax.hist(fake_scores, bins=40, alpha=0.7, color="#ef5350", label="Fake")
    ax.axvline(0.5, color="white", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("Fake Probability Score", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Score Distribution", color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white")

    filename = f"{prefix}_score_distribution.png" if prefix else "score_distribution.png"

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, facecolor="#0d0d1a")
    plt.close()


def _plot_threshold_analysis(y_true, y_score, out_dir: Path, prefix=""):
    """Plot Accuracy, F1, Precision, Recall vs classification threshold."""
    thresholds = np.linspace(0.1, 0.9, 80)
    metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_score]
        metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["f1"].append(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d0d1a")
    ax.set_facecolor("#1a1a2e")
    colors = {"accuracy": "#7c6ff7", "f1": "#ffa726", "precision": "#66bb6a", "recall": "#ef5350"}
    for name, vals in metrics.items():
        ax.plot(thresholds, vals, label=name.capitalize(), color=colors[name], lw=1.8)

    ax.axvline(0.5, color="white", linestyle="--", lw=1, alpha=0.4, label="Default (0.5)")
    ax.set_xlabel("Threshold", color="white")
    ax.set_ylabel("Score", color="white")
    ax.set_title("Metrics vs. Classification Threshold", color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")

    filename = f"{prefix}_threshold_analysis.png" if prefix else "threshold_analysis.png"

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120, facecolor="#0d0d1a")
    plt.close()
    logger.info("Saved threshold_analysis.png")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate deepfake detection pipeline")
    p.add_argument("--data_dir",    required=True,
                   help="Directory with real/ and fake/ subdirs of test videos")
    p.add_argument("--video_model", default="data/models/video_model.pth")
    p.add_argument("--audio_model", default="data/models/audio_model.pth")
    p.add_argument("--output_dir",  default="eval_results")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit evaluation to N samples (for quick sanity checks)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Collect test videos
    test_videos = []
    for label, cls in [(0, "real"), (1, "fake")]:
        for ext in ["*.mp4", "*.avi", "*.mov"]:
            test_videos.extend([(p, label) for p in (data_dir / cls).glob(ext)])

    if args.max_samples:
        import random; random.shuffle(test_videos)
        test_videos = test_videos[:args.max_samples]

    logger.info(f"Test set: {len(test_videos)} videos")

    # Load models
    video_model = VideoInference(Path(args.video_model), args.device)
    audio_model = AudioInference(Path(args.audio_model), args.device)

    metrics = evaluate_pipeline(test_videos, video_model, audio_model, output_dir, args.device)
    print("\n" + "─" * 50)
    print("EVALUATION SUMMARY")
    print("─" * 50)
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    print("─" * 50)
    print(f"\nPlots saved to: {output_dir}")