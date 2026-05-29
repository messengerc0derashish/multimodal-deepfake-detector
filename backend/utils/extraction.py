"""
utils/extraction.py — Input pipeline for extracting frames, audio, and transcript.

Pipeline:
  video file → frames (OpenCV) + audio WAV (ffmpeg) + transcript (Whisper)
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import librosa
from loguru import logger

from backend.config import settings

WHISPER_AVAILABLE = False


# ─── Frame Extraction ─────────────────────────────────────────────────────────

def extract_frames(
    video_path: str | Path,
    sample_rate: int = settings.FRAME_SAMPLE_RATE,
    max_frames: Optional[int] = settings.MAX_FRAMES,
) -> List[np.ndarray]:
    """
    Extract frames from a video at `sample_rate` (every N-th frame).
    Returns list of BGR numpy arrays (H, W, 3).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(f"Video: {total_frames} frames @ {fps:.1f} fps")

    frames: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
    return frames


def get_video_metadata(video_path: str | Path) -> dict:
    """Return basic metadata: fps, duration, resolution."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_s": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return meta


# ─── Audio Extraction ─────────────────────────────────────────────────────────

def extract_audio(
    video_path: str | Path,
    output_wav: Optional[str | Path] = None,
    sample_rate: int = settings.AUDIO_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Extract audio track from video using ffmpeg.
    Returns (waveform_array, sample_rate).
    """
    video_path = Path(video_path)
    if output_wav is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_wav = Path(tmp.name)
    else:
        output_wav = Path(output_wav)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                          # no video
        "-acodec", "pcm_s16le",         # 16-bit PCM
        "-ar", str(sample_rate),        # resample
        "-ac", "1",                     # mono
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"ffmpeg audio extraction warning: {result.stderr[-300:]}")

    if not output_wav.exists() or output_wav.stat().st_size == 0:
        logger.warning("No audio stream found — returning silent signal")
        duration = get_video_metadata(video_path).get("duration_s", 5.0)
        waveform = np.zeros(int(sample_rate * min(duration, settings.AUDIO_MAX_DURATION)))
        return waveform, sample_rate

    waveform, sr = librosa.load(str(output_wav), sr=sample_rate, mono=True,
                                duration=settings.AUDIO_MAX_DURATION)
    logger.info(f"Audio extracted: {len(waveform)/sr:.1f}s @ {sr} Hz")
    return waveform, sr


def compute_mfcc(
    waveform: np.ndarray,
    sr: int = settings.AUDIO_SAMPLE_RATE,
    n_mfcc: int = settings.AUDIO_N_MFCC,
    hop_length: int = settings.AUDIO_HOP_LENGTH,
    n_fft: int = settings.AUDIO_N_FFT,
    max_len: int = 256,
) -> np.ndarray:
    """
    Compute MFCC features from a waveform.
    Returns array of shape (n_mfcc, max_len) — zero-padded / truncated.
    """
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc,
        hop_length=hop_length, n_fft=n_fft,
    )
    # Normalise
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    # Fixed length
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.astype(np.float32)


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = settings.AUDIO_SAMPLE_RATE,
    n_mels: int = 128,
    hop_length: int = settings.AUDIO_HOP_LENGTH,
    max_len: int = 256,
) -> np.ndarray:
    """
    Compute log-mel spectrogram — (n_mels, max_len).
    """
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    if log_mel.shape[1] < max_len:
        log_mel = np.pad(log_mel, ((0, 0), (0, max_len - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :max_len]

    return log_mel.astype(np.float32)


# ─── Transcript Extraction ────────────────────────────────────────────────────


def extract_transcript(
    video_path: str | Path,
    model_size: str = "base",
) -> dict:
    """
    Transcribe video audio using faster-whisper.
    Returns dict with 'text', 'segments', and 'language' keys.
    """
  try:
        model = get_whisper_model(model_size)
    
        # faster-whisper returns a generator — consume it fully
        segments_gen, info = model.transcribe(str(video_path), beam_size=5)
    
        segments = []
        full_text = ""
        for seg in segments_gen:
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg.text.strip(),
            })
            full_text += seg.text + " "
    
        full_text = full_text.strip()
        logger.info(f"Transcript ({info.language}): {full_text[:120]}...")
        return {
            "text":     full_text,
            "segments": segments,
            "language": info.language,
        }
  except Exception as e:
        return {
            "text": '',
            "segments": '',
            "language": '',
        }
