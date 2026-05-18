"""
config.py — Central configuration for the Deepfake Detection System.
All paths, hyperparameters, and runtime settings live here.
"""

from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Deepfake Detection System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Paths ─────────────────────────────────────────────────────────────────
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    MODEL_DIR: Path = BASE_DIR / "data" / "models"
    LOG_DIR: Path = BASE_DIR / "logs"

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE: str = "cuda"            # "cuda" | "cpu" | "mps"
    NUM_WORKERS: int = 4

    # ── Video extraction ──────────────────────────────────────────────────────
    FRAME_SAMPLE_RATE: int = 10     # sample every N-th frame
    MAX_FRAMES: int = 64            # cap per video
    FACE_IMAGE_SIZE: int = 224      # input size for face model

    # ── Audio extraction ──────────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_N_MFCC: int = 40
    AUDIO_HOP_LENGTH: int = 512
    AUDIO_N_FFT: int = 1024
    AUDIO_MAX_DURATION: float = 30.0   # seconds

    # ── Model weights ─────────────────────────────────────────────────────────
    VIDEO_MODEL_PATH: Path = BASE_DIR / "data" / "models" / "video_model.pth"
    AUDIO_MODEL_PATH: Path = BASE_DIR / "data" / "models" / "audio_model.pth"
    TEXT_MODEL_NAME: str = "distilbert-base-uncased"

    # ── Fusion weights (must sum to 1) ────────────────────────────────────────
    VIDEO_WEIGHT: float = 0.50
    AUDIO_WEIGHT: float = 0.35
    TEXT_WEIGHT: float = 0.15

    # ── Detection thresholds ──────────────────────────────────────────────────
    FAKE_THRESHOLD: float = 0.50    # score >= this → FAKE
    HIGH_CONFIDENCE: float = 0.80
    FRAME_FAKE_THRESHOLD: float = 0.60
    VIDEO_FAKE_FRAME_RATIO_THRESHOLD: float = 0.25
    VIDEO_SUSPICIOUS_FRAME_RATIO_THRESHOLD: float = 0.15

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE: int = 32
    EPOCHS: int = 30
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 5               # early stopping
    GRAD_CLIP: float = 1.0

    # ── API ───────────────────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 500
    ALLOWED_EXTENSIONS: set = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist at import time
for _dir in [settings.UPLOAD_DIR, settings.MODEL_DIR, settings.LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
