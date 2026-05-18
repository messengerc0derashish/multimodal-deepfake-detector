"""
backend/worker.py — Celery worker for async / batch deepfake analysis.

Usage:
    # Start worker
    celery -A backend.worker worker --loglevel=info --concurrency=1

    # Submit batch job from Python
    from backend.worker import analyze_video_task
    result = analyze_video_task.delay("/path/to/video.mp4")
    print(result.get(timeout=300))
"""

_pipeline = None

def get_pipeline_lazy():
    global _pipeline

    if _pipeline is None:
        from backend.inference_pipeline import get_pipeline

        logger.info("Loading inference pipeline...")
        _pipeline = get_pipeline()

    return _pipeline

from celery import Celery
from loguru import logger
from pathlib import Path

from backend.config import settings

celery_app = Celery(
    "deepfake_worker",
    broker=settings.REDIS_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,              # 1 hour
    task_soft_time_limit=600,         # 10 min warning
    task_time_limit=900,              # 15 min hard kill
    worker_prefetch_multiplier=1,     # one task at a time per worker
)


@celery_app.task(bind=True, name="analyze_video")
def analyze_video_task(self, video_path: str) -> dict:
    """
    Celery task: full deepfake analysis of a video file.
    Returns serialisable result dict.
    """
    from backend.inference_pipeline import get_pipeline
    from dataclasses import asdict

    logger.info(f"[Task {self.request.id}] Analysing: {video_path}")
    self.update_state(state="PROGRESS", meta={"step": "loading pipeline"})

    try:
        pipeline = get_pipeline_lazy()
        self.update_state(state="PROGRESS", meta={"step": "running inference"})
        result = pipeline.run(video_path)
        result_dict = asdict(result)
        logger.success(f"[Task {self.request.id}] Done → {result.label}")
        return result_dict
    except Exception as exc:
        logger.exception(f"[Task {self.request.id}] Failed: {exc}")
        raise self.retry(exc=exc, countdown=5, max_retries=2)


@celery_app.task(name="batch_analyze")
def batch_analyze_task(video_paths: list[str]) -> list[dict]:
    """Process a list of videos and return all results."""
    from backend.inference_pipeline import get_pipeline
    from dataclasses import asdict

    pipeline = get_pipeline_lazy()
    results = []
    for path in video_paths:
        try:
            r = pipeline.run(path)
            results.append({"path": path, **asdict(r)})
        except Exception as e:
            results.append({"path": path, "error": str(e)})
    return results
