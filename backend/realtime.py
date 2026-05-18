"""
backend/realtime.py — Real-time deepfake detection from webcam or RTSP stream.

Features:
  - Live face detection and analysis every N frames
  - Overlaid verdict + confidence score on video feed
  - Audio capture (via sounddevice) for audio-side inference
  - Press Q to quit, S to save a snapshot

Usage:
    # Webcam (default device 0)
    python -m backend.realtime --source 0

    # RTSP stream
    python -m backend.realtime --source rtsp://192.168.1.10:554/stream

    # API mode: stream frames to backend (no local GPU needed)
    python -m backend.realtime --source 0 --api_mode
"""

import argparse
import time
import threading
import queue
import tempfile
from pathlib import Path
from typing import Optional
from collections import deque

import cv2
import numpy as np
from loguru import logger


# ── Colour palette ─────────────────────────────────────────────────────────────
FAKE_BGR  = (80, 80, 239)    # red-ish in BGR
REAL_BGR  = (100, 187, 101)  # green in BGR
WARN_BGR  = (50, 150, 240)   # amber in BGR
WHITE_BGR = (220, 220, 220)
DARK_BGR  = (20, 20, 35)


def draw_overlay(
    frame: np.ndarray,
    label: str,
    fake_prob: float,
    video_score: Optional[float],
    audio_score: Optional[float],
    fps: float,
) -> np.ndarray:
    """Draw verdict box and score bars on a frame."""
    h, w = frame.shape[:2]

    label_color = {
        "FAKE": FAKE_BGR,
        "REAL": REAL_BGR,
        "UNCERTAIN": WARN_BGR,
    }.get(label, WHITE_BGR)

    # ── Main verdict box ──────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 160), DARK_BGR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, label, (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, label_color, 2)
    cv2.putText(frame, f"Fake: {fake_prob:.1%}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE_BGR, 1)

    # ── Score bars ────────────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 20, 105, 200, 12

    def _bar(y, score, label_text, color):
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (60, 60, 80), -1)
        filled = int(bar_w * score)
        cv2.rectangle(frame, (bar_x, y), (bar_x + filled, y + bar_h), color, -1)
        cv2.putText(frame, f"{label_text}: {score:.0%}", (bar_x + bar_w + 8, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_BGR, 1)

    if video_score is not None:
        _bar(bar_y, video_score, "Video", FAKE_BGR if video_score > 0.5 else REAL_BGR)
    if audio_score is not None:
        _bar(bar_y + 20, audio_score, "Audio", FAKE_BGR if audio_score > 0.5 else REAL_BGR)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE_BGR, 1)
    cv2.putText(frame, "Press Q to quit | S to snapshot", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    return frame


class RealtimeDetector:
    """
    Runs deepfake detection on a live video stream.
    Analysis is done in a background thread every ANALYSE_EVERY frames
    so the display loop stays smooth.
    """

    ANALYSE_EVERY = 30       # frames between full analyses
    AUDIO_SECONDS = 3.0      # seconds of audio to capture per analysis

    def __init__(
        self,
        source: str | int,
        device: str = "cpu",
        api_mode: bool = False,
        api_url: str = "http://localhost:8000",
    ):
        self.source   = source
        self.device   = device
        self.api_mode = api_mode
        self.api_url  = api_url

        # Result state (updated by background thread)
        self._label      = "ANALYSING..."
        self._fake_prob  = 0.0
        self._video_score = None
        self._audio_score = None
        self._lock       = threading.Lock()

        # Frame queue for background analysis
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = True

        if not api_mode:
            self._load_local_models()

    def _load_local_models(self):
        from models.video.video_model import VideoInference
        from models.audio.audio_model import AudioInference
        logger.info("Loading models for local inference...")
        self.video_model = VideoInference(settings.VIDEO_MODEL_PATH, self.device)
        self.audio_model = AudioInference(settings.AUDIO_MODEL_PATH, self.device)

    def _analyse_worker(self):
        """Background thread: dequeue frames and run inference."""
        from models.fusion.fusion import FusionOrchestrator, ModalityResult
        fusion = FusionOrchestrator("weighted")

        while self._running:
            try:
                frames = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if self.api_mode:
                    result = self._api_analyse(frames)
                else:
                    result = self._local_analyse(frames, fusion)

                with self._lock:
                    self._label       = result.get("label", "UNCERTAIN")
                    self._fake_prob   = result.get("fake_probability", 0.5)
                    self._video_score = result.get("video_score")
                    self._audio_score = result.get("audio_score")

            except Exception as e:
                logger.warning(f"Analysis error: {e}")

    def _local_analyse(self, frames: list, fusion) -> dict:
        from models.fusion.fusion import ModalityResult
        import sounddevice as sd

        v_out = self.video_model.predict(frames)

        # Capture audio
        try:
            audio = sd.rec(
                int(self.AUDIO_SECONDS * 16000),
                samplerate=16000, channels=1, dtype="float32",
            )
            sd.wait()
            waveform = audio.flatten()
        except Exception:
            waveform = np.zeros(int(16000 * self.AUDIO_SECONDS), dtype=np.float32)

        a_out = self.audio_model.predict(waveform, 16000)

        result = fusion.combine([
            ModalityResult("video", v_out["fake_probability"],
                           1.0 if v_out.get("face_detected") else 0.5),
            ModalityResult("audio", a_out["fake_probability"], 1.0),
        ])
        return {
            "label": result.label,
            "fake_probability": result.fake_probability,
            "video_score": v_out["fake_probability"],
            "audio_score": a_out["fake_probability"],
        }

    def _api_analyse(self, frames: list) -> dict:
        """Send frames to backend API (api_mode=True)."""
        import requests, tempfile, cv2

        # Write frames to temp video
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
        )
        for f in frames:
            writer.write(f)
        writer.release()

        with open(tmp.name, "rb") as f:
            r = requests.post(
                f"{self.api_url}/api/analyze_sync",
                files={"file": ("realtime.mp4", f, "video/mp4")},
                timeout=60,
            )
        if r.ok:
            return r.json()
        return {"label": "UNCERTAIN", "fake_probability": 0.5}

    def run(self):
        """Main loop: capture → display → periodic analysis."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"Cannot open source: {self.source}")
            return

        logger.info(f"Stream opened: {self.source}")

        # Start background worker
        worker = threading.Thread(target=self._analyse_worker, daemon=True)
        worker.start()

        frame_count  = 0
        buffer       = deque(maxlen=30)   # hold last 30 frames for analysis
        fps_times    = deque(maxlen=30)
        snapshot_dir = Path("snapshots")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream ended or cannot read frame")
                    break

                buffer.append(frame.copy())
                frame_count += 1
                fps_times.append(time.perf_counter())

                # Enqueue frames for analysis
                if frame_count % self.ANALYSE_EVERY == 0:
                    if not self._frame_queue.full():
                        self._frame_queue.put(list(buffer))

                # Compute FPS
                fps = 0.0
                if len(fps_times) > 1:
                    fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-9)

                # Draw overlay with latest result
                with self._lock:
                    label      = self._label
                    fake_prob  = self._fake_prob
                    vid_score  = self._video_score
                    aud_score  = self._audio_score

                display = draw_overlay(frame, label, fake_prob, vid_score, aud_score, fps)
                cv2.imshow("Deepfake Detector — Real-time", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    snapshot_dir.mkdir(exist_ok=True)
                    path = snapshot_dir / f"snap_{int(time.time())}.jpg"
                    cv2.imwrite(str(path), display)
                    logger.info(f"Snapshot saved: {path}")

        finally:
            self._running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Detector stopped")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.config import settings

    p = argparse.ArgumentParser(description="Real-time deepfake detection")
    p.add_argument("--source",   default=0,
                   help="Camera index (0,1,...) or stream URL")
    p.add_argument("--device",   default="cpu")
    p.add_argument("--api_mode", action="store_true",
                   help="Send frames to backend API instead of running locally")
    p.add_argument("--api_url",  default="http://localhost:8000")
    args = p.parse_args()

    source = args.source
    try:
        source = int(source)   # convert "0" → 0
    except (ValueError, TypeError):
        pass

    detector = RealtimeDetector(
        source=source,
        device=args.device,
        api_mode=args.api_mode,
        api_url=args.api_url,
    )
    detector.run()