"""
backend/inference_pipeline.py — End-to-end inference orchestrator.

Ties together:
  1. Extraction  (frames + audio + transcript)
  2. Video model inference
  3. Audio model inference
  4. Text/lip-sync inference
  5. Multimodal fusion
  6. Explainability generation

Returns a fully structured AnalysisResult.
"""

import base64
import time
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from backend.config import settings
from backend.utils.extraction import (
    extract_frames, extract_audio, extract_transcript, get_video_metadata,
)
from models.video.video_model import VideoInference
from models.audio.audio_model import AudioInference
from models.text.text_model import TextInference
from models.fusion.fusion import FusionOrchestrator, ModalityResult
from models.explainability import plot_audio_analysis, plot_modality_breakdown


# ─── Result Schema ────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    # High-level verdict
    label:            str              # "REAL" | "FAKE" | "UNCERTAIN"
    fake_probability: float
    confidence:       str              # "HIGH" | "MEDIUM" | "LOW"

    # Per-modality breakdown
    video_score:  Optional[float]      = None
    audio_score:  Optional[float]      = None
    text_score:   Optional[float]      = None
    frame_analysis: Optional[dict]     = None
    modality_details: dict             = field(default_factory=dict)

    # Explanations
    explanation:  list                 = field(default_factory=list)

    # Visualisations (base64 PNG)
    audio_plot:   Optional[str]        = None
    breakdown_plot: Optional[str]      = None
    suspicious_frames: list            = field(default_factory=list)

    # Metadata
    video_metadata: dict               = field(default_factory=dict)
    processing_time_s: float           = 0.0
    error:            Optional[str]    = None


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Singleton-friendly pipeline.  Loads models once at startup.
    """

    def __init__(self, fusion_strategy: str = "weighted"):
        device = settings.DEVICE
        logger.info(f"Initialising pipeline on device: {device}")

        self.video_model = VideoInference(settings.VIDEO_MODEL_PATH, device)
        self.audio_model = AudioInference(settings.AUDIO_MODEL_PATH, device)
        self.text_model  = TextInference(
            model_name=settings.TEXT_MODEL_NAME,
            device=device,
        )
        self.fusion = FusionOrchestrator(strategy=fusion_strategy)
        logger.success("Pipeline ready")

    def run(self, video_path: str | Path) -> AnalysisResult:
        """
        Full analysis of a video file.
        Returns AnalysisResult with all scores and explanations.
        """
        start = time.perf_counter()
        video_path = Path(video_path)

        if not video_path.exists():
            return AnalysisResult(
                label="ERROR", fake_probability=0.5,
                confidence="LOW", error=f"File not found: {video_path}",
            )

        try:
            return self._run_pipeline(video_path, start)
        except Exception as exc:
            logger.exception(f"Pipeline error: {exc}")
            return AnalysisResult(
                label="ERROR", fake_probability=0.5,
                confidence="LOW",
                processing_time_s=time.perf_counter() - start,
                error=str(exc),
            )

    def _run_pipeline(self, video_path: Path, start: float) -> AnalysisResult:
        # ── Step 1: Metadata ──────────────────────────────────────────────────
        meta = get_video_metadata(video_path)
        duration = meta.get("duration_s", 0.0)
        logger.info(f"Video metadata: {meta}")

        # ── Step 2: Extract frames ────────────────────────────────────────────
        logger.info("Extracting frames...")
        frames = extract_frames(video_path, sample_rate=1, max_frames=None)

        # ── Step 3: Try extracting audio (optional) ───────────────────────────
        logger.info("Extracting audio...")
        waveform      = None
        sr            = None
        audio_available = False

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            waveform, sr = extract_audio(video_path, output_wav=wav_path)

            # Check if audio is actually meaningful (not silent/empty)
            import numpy as np
            if (
                waveform is not None
                and len(waveform) > sr * 0.5          # at least 0.5s
                and np.sqrt(np.mean(waveform ** 2)) > 0.001  # not silent
            ):
                audio_available = True
                logger.info(f"Audio available: {len(waveform)/sr:.1f}s")
            else:
                logger.warning("Audio too short or silent — skipping audio/text analysis")

        except Exception as e:
            logger.warning(f"Audio extraction failed: {e} — skipping audio/text analysis")

        # ── Step 4: Try transcript (only if audio available) ──────────────────
        transcript = {"text": "", "segments": [], "language": "en"}
        if audio_available:
            logger.info("Extracting transcript...")
            try:
                transcript = extract_transcript(video_path, model_size="base")
            except Exception as e:
                logger.warning(f"Transcript failed: {e} — skipping text analysis")

        # ── Step 5: Video inference (always runs) ─────────────────────────────
        logger.info("Running video model...")
        video_out = self.video_model.predict(frames)
        video_result = ModalityResult(
            modality="video",
            fake_probability=video_out["fake_probability"],
            confidence=1.0 if video_out.get("face_detected") else 0.5,
            explanation=video_out.get("explanation", []),
            raw=video_out,
        )
        modality_results = [video_result]

        # ── Step 6: Audio inference (only if audio available) ─────────────────
        audio_out  = None
        audio_result = None
        if audio_available:
            logger.info("Running audio model...")
            try:
                audio_out = self.audio_model.predict(waveform, sr)
                audio_result = ModalityResult(
                    modality="audio",
                    fake_probability=audio_out["fake_probability"],
                    confidence=1.0,
                    explanation=audio_out.get("explanation", []),
                    raw=audio_out,
                )
                modality_results.append(audio_result)
            except Exception as e:
                logger.warning(f"Audio model failed: {e} — skipping")

        # ── Step 7: Text inference (only if audio available) ──────────────────
        text_out   = None
        text_result = None
        if audio_available and waveform is not None:
            logger.info("Running text model...")
            try:
                text_out = self.text_model.predict(
                    transcript, waveform, sr, duration
                )
                text_result = ModalityResult(
                    modality="text",
                    fake_probability=text_out["fake_probability"],
                    confidence=0.9 if transcript["text"] else 0.3,
                    explanation=text_out.get("explanation", []),
                    raw=text_out,
                )
                modality_results.append(text_result)
            except Exception as e:
                logger.warning(f"Text model failed: {e} — skipping")

        # ── Step 8: Fusion (uses whatever modalities are available) ───────────
        logger.info(f"Fusing {len(modality_results)} modalities: "
                    f"{[r.modality for r in modality_results]}")
        fusion_out = self.fusion.combine(modality_results)
        fusion_out = self._apply_frame_score_formula(
            fusion_out,
            video_out,
            audio_result,
            text_result,
        )

        # ── Step 9: Explainability ────────────────────────────────────────────
        audio_plot    = None
        breakdown_plot = None
        try:
            if audio_available and audio_out:
                audio_plot = plot_audio_analysis(
                    waveform, sr,
                    suspicious_segments=audio_out.get("suspicious_segments", []),
                    fake_probability=audio_out["fake_probability"],
                )
            breakdown_plot = plot_modality_breakdown(
                fusion_out.modality_scores,
                fusion_out.fake_probability,
            )
        except Exception as e:
            logger.warning(f"Visualisation error: {e}")

        elapsed = round(time.perf_counter() - start, 2)
        suspicious_frames = self._build_suspicious_frames(
            frames,
            video_out.get("frame_scores", []),
        )
        logger.success(
            f"Analysis complete in {elapsed}s → {fusion_out.label} "
            f"({fusion_out.fake_probability:.1%}) "
            f"| modalities used: {[r.modality for r in modality_results]}"
        )

        return AnalysisResult(
            label=fusion_out.flag,
            fake_probability=fusion_out.fake_probability,
            confidence=fusion_out.confidence,
            video_score=video_result.fake_probability,
            audio_score=audio_result.fake_probability if audio_result else None,
            text_score=text_result.fake_probability   if text_result  else None,
            frame_analysis=video_out.get("frame_analysis"),
            modality_details=fusion_out.modality_scores,
            explanation=fusion_out.explanation + (
                ["Audio/text analysis skipped — no audio stream detected"]
                if not audio_available else []
            ),
            audio_plot=audio_plot,
            breakdown_plot=breakdown_plot,
            suspicious_frames=suspicious_frames,
            video_metadata=meta,
            processing_time_s=elapsed,
        )


# ─── Module-level singleton ───────────────────────────────────────────────────
    def _build_suspicious_frames(self, frames: list[np.ndarray], frame_scores: list[dict]) -> list[dict]:
        if not frames or not frame_scores:
            return []

        ranked = sorted(
            [
                {
                    "frame_idx": item["frame_idx"],
                    "score": float(item["score"]),
                }
                for item in frame_scores
                if 0 <= item.get("frame_idx", -1) < len(frames)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )

        selected = [
            item for item in ranked
            if item["score"] >= settings.FRAME_FAKE_THRESHOLD
        ][:4]
        if not selected:
            selected = ranked[:3]

        suspicious_frames = []
        for item in selected:
            frame = frames[item["frame_idx"]]
            success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                continue

            suspicious_frames.append(
                {
                    "frame_idx": item["frame_idx"],
                    "score": round(item["score"], 4),
                    "image": base64.b64encode(encoded.tobytes()).decode("utf-8"),
                }
            )

        return suspicious_frames

    def _apply_frame_score_formula(self, fusion_out, video_out: dict, audio_result=None, text_result=None):
        frame_analysis = video_out.get("frame_analysis") or {}
        ratio_verdict = frame_analysis.get("frame_ratio_verdict")
        if ratio_verdict not in {"FAKE", "SUSPICIOUS", "REAL"}:
            return fusion_out

        frame_component = frame_analysis.get("frame_component", 0.0)
        if audio_result is None:
            final_fake_score = frame_component
            score_note = "audio unavailable, using frame component only"
            frame_analysis["score_formula"] = {
                "frame_weight": 1.0,
                "audio_weight": 0.0,
                "text_weight": 0.0,
                "frame_component": round(frame_component, 4),
                "audio_score": None,
                "text_score": None,
                "audio_available": False,
                "text_available": False,
                "final_fake_score": round(float(np.clip(final_fake_score, 0.0, 1.0)), 4),
            }
        else:
            audio_score = audio_result.fake_probability
            text_score = text_result.fake_probability if text_result else 0.5
            final_fake_score = (
                frame_component * 0.7
                + audio_score * 0.15
                + text_score * 0.15
            )
            score_note = (
                f"frame component {frame_component:.1%}, "
                f"audio {audio_score:.1%}, text {text_score:.1%}"
            )
            frame_analysis["score_formula"] = {
                "frame_weight": 0.7,
                "audio_weight": 0.15,
                "text_weight": 0.15,
                "frame_component": round(frame_component, 4),
                "audio_score": round(audio_score, 4),
                "text_score": round(text_score, 4),
                "audio_available": True,
                "text_available": text_result is not None,
                "final_fake_score": round(float(np.clip(final_fake_score, 0.0, 1.0)), 4),
            }

        fusion_out.fake_probability = round(float(np.clip(final_fake_score, 0.0, 1.0)), 4)
        active_verdict = self._resolve_final_verdict(
            ratio_verdict,
            fusion_out.fake_probability,
            audio_result.fake_probability if audio_result else None,
            text_result.fake_probability if text_result else None,
        )
        fusion_out.label = active_verdict
        fusion_out.flag = active_verdict
        if active_verdict == "FAKE":
            fusion_out.confidence = (
                "HIGH"
                if fusion_out.fake_probability >= settings.HIGH_CONFIDENCE
                else "MEDIUM"
            )
        elif active_verdict == "SUSPICIOUS":
            fusion_out.confidence = "LOW"
        else:
            fusion_out.confidence = "HIGH"

        fusion_out.explanation = [
            line for line in fusion_out.explanation
            if not str(line).startswith("Final score:")
        ]
        fusion_out.explanation.insert(
            0,
            f"Active verdict: {active_verdict} "
            f"({frame_analysis.get('fake_frame_ratio', 0.0):.1%} fake frames); "
            f"decision score from formula: {fusion_out.fake_probability:.1%} ({score_note})",
        )
        return fusion_out

    def _resolve_final_verdict(
        self,
        ratio_verdict: str,
        final_fake_score: float,
        audio_score: float | None,
        text_score: float | None,
    ) -> str:
        if ratio_verdict == "FAKE":
            return "FAKE"
        if final_fake_score >= settings.FAKE_THRESHOLD:
            return "FAKE"
        if audio_score is not None and audio_score >= 0.85:
            return "SUSPICIOUS"
        if text_score is not None and text_score >= 0.85:
            return "SUSPICIOUS"
        return ratio_verdict

    def _has_audio_signal(self, waveform: np.ndarray) -> bool:
        if waveform is None or len(waveform) == 0:
            return False
        return bool(np.max(np.abs(waveform)) > 1e-6)


_pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    """Return (or create) the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline
