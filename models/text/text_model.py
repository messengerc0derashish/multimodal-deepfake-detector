"""
Text / transcript deepfake analysis helpers.

This module combines:
1. Transformer-based transcript scoring when local model files are available.
2. Offline-safe heuristic transcript checks.
3. Heuristic lip-sync mismatch scoring.
"""

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class TextDeepfakeDetector(nn.Module):
    """
    DistilBERT-based binary classifier for transcript anomaly detection.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logit = self.classifier(cls).squeeze(-1)
        return {"logit": logit, "prob": torch.sigmoid(logit)}


class LipSyncAnalyser:
    """
    Simple lip-sync mismatch detector based on speech segments and audio energy.
    """

    def analyse(
        self,
        transcript_segments: List[dict],
        video_duration_s: float,
        audio_waveform: np.ndarray,
        audio_sr: int,
    ) -> dict:
        if not transcript_segments or video_duration_s <= 0:
            return {
                "mismatch_score": 0.0,
                "suspicious_intervals": [],
                "explanation": ["Insufficient data for lip-sync analysis"],
            }

        spoken = [
            (segment["start"], segment["end"])
            for segment in transcript_segments
            if segment.get("text", "").strip()
        ]

        mismatch_count = 0
        total_speech = 0
        suspicious = []

        for start_s, end_s in spoken:
            total_speech += 1
            segment_waveform = self._slice_audio(audio_waveform, audio_sr, start_s, end_s)
            rms = (
                float(np.sqrt(np.mean(segment_waveform ** 2)))
                if len(segment_waveform) > 0
                else 0.0
            )
            if rms < 0.005:
                mismatch_count += 1
                suspicious.append(
                    {
                        "start_s": round(start_s, 2),
                        "end_s": round(end_s, 2),
                        "type": "speech_with_no_audio",
                    }
                )

        mismatch_score = mismatch_count / max(total_speech, 1)
        mismatch_score = min(mismatch_score * 2.0, 1.0)

        return {
            "mismatch_score": round(mismatch_score, 4),
            "suspicious_intervals": suspicious[:10],
            "explanation": self._build_explanation(mismatch_score, suspicious),
        }

    def _slice_audio(
        self,
        waveform: np.ndarray,
        sr: int,
        start_s: float,
        end_s: float,
    ) -> np.ndarray:
        start = int(start_s * sr)
        end = int(end_s * sr)
        return waveform[start:end] if end <= len(waveform) else waveform[start:]

    def _build_explanation(self, score: float, suspicious: list) -> List[str]:
        reasons = []
        if score > 0.6:
            reasons.append(f"Significant lip-sync mismatch detected (score={score:.2f})")
        elif score > 0.3:
            reasons.append(f"Possible lip-sync inconsistency (score={score:.2f})")
        else:
            reasons.append(f"Lip-sync appears consistent (score={score:.2f})")

        if suspicious:
            reasons.append(f"{len(suspicious)} speech segment(s) have no audio energy")

        return reasons


class TextInference:
    """
    High-level wrapper combining transcript heuristics and lip-sync analysis.

    If the transformer assets are not available locally, this class stays usable
    in offline mode and falls back to heuristic-only transcript scoring.
    """

    MAX_TOKENS = 512

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.tokenizer = None
        self.nlp_model = None
        self.transformer_available = False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )
            self.nlp_model = TextDeepfakeDetector(model_name=model_name)

            if model_path and Path(model_path).exists():
                state = torch.load(model_path, map_location=device)
                self.nlp_model.load_state_dict(state)
                logger.info(f"Text model loaded from {model_path}")
            else:
                logger.warning("No trained text weights - using random init (for demo)")

            self.nlp_model.to(device).eval()
            self.transformer_available = True
        except Exception as exc:
            logger.warning(
                "Text transformer unavailable locally; using offline heuristic mode only: {}",
                exc,
            )

        self.lip_sync = LipSyncAnalyser()

    @torch.no_grad()
    def predict(
        self,
        transcript: dict,
        audio_waveform: np.ndarray,
        audio_sr: int,
        video_duration_s: float,
    ) -> dict:
        text = transcript.get("text", "").strip()
        segments = transcript.get("segments", [])

        nlp_result = self._classify_text(text)
        lip_result = self.lip_sync.analyse(
            segments,
            video_duration_s,
            audio_waveform,
            audio_sr,
        )

        combined_score = (
            0.6 * nlp_result["fake_probability"] + 0.4 * lip_result["mismatch_score"]
        )

        return {
            "fake_probability": round(combined_score, 4),
            "nlp_score": nlp_result["fake_probability"],
            "lipsync_score": lip_result["mismatch_score"],
            "transcript_length": len(text.split()),
            "suspicious_intervals": lip_result["suspicious_intervals"],
            "explanation": nlp_result["explanation"] + lip_result["explanation"],
        }

    def _classify_text(self, text: str) -> dict:
        if not text:
            return {"fake_probability": 0.5, "explanation": ["No transcript available"]}

        heuristic_score, heuristic_reasons = self._heuristic_checks(text)

        if not self.transformer_available or self.tokenizer is None or self.nlp_model is None:
            explanation = heuristic_reasons[:] if heuristic_reasons else []
            explanation.append(
                "Transformer text model unavailable offline; using heuristic transcript checks only"
            )
            return {
                "fake_probability": round(heuristic_score, 4),
                "explanation": explanation,
            }

        enc = self.tokenizer(
            text,
            max_length=self.MAX_TOKENS,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {key: value.to(self.device) for key, value in enc.items()}
        out = self.nlp_model(**enc)
        model_score = float(out["prob"][0].cpu())

        final_score = 0.4 * heuristic_score + 0.6 * model_score
        return {
            "fake_probability": round(final_score, 4),
            "explanation": heuristic_reasons,
        }

    def _heuristic_checks(self, text: str) -> tuple[float, List[str]]:
        score = 0.0
        reasons = []

        words = text.split()
        if len(words) == 0:
            return 0.5, ["Empty transcript"]

        punct_ratio = len(re.findall(r"[.,!?;]", text)) / max(len(words), 1)
        if punct_ratio > 0.5:
            score += 0.2
            reasons.append("Unusually high punctuation density in transcript")

        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            score += 0.3
            reasons.append("High word repetition detected in transcript")

        if len(words) < 5:
            reasons.append("Very short transcript")
        elif len(words) > 500:
            score += 0.1
            reasons.append("Unusually long transcript")

        return min(score, 1.0), reasons
