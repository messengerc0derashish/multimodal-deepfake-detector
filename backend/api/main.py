"""
backend/api/main.py — FastAPI application entry point.

Endpoints:
  POST /api/upload          → upload video, get job_id
  POST /api/analyze/{job_id}→ trigger analysis (sync or async)
  GET  /api/result/{job_id} → poll result
  GET  /api/health          → health check
  GET  /api/stream          → server-sent events for real-time progress

CORS is enabled for the React frontend at localhost:3000 / 5173.
"""

import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from loguru import logger

from backend.config import settings
from backend.inference_pipeline import get_pipeline, AnalysisResult


# ── Logging setup ─────────────────────────────────────────────────────────────
logger.add(
    settings.LOG_DIR / "api.log",
    rotation="100 MB", retention="30 days",
    level=settings.LOG_LEVEL, serialize=False,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multimodal Deepfake Detection API",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://multimodal-deepfake-detector.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (use Redis in production) ─────────────────────────────
_jobs: dict[str, dict] = {}


# ─── Schema ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    size_mb: float
    message: str


class AnalysisResponse(BaseModel):
    job_id:           str
    status:           str       # "pending" | "processing" | "done" | "error"
    label:            Optional[str]
    fake_probability: Optional[float]
    confidence:       Optional[str]
    video_score:      Optional[float]
    audio_score:      Optional[float]
    text_score:       Optional[float]
    frame_analysis:   Optional[dict]
    modality_details: Optional[dict]
    explanation:      Optional[list]
    audio_plot:       Optional[str]
    breakdown_plot:   Optional[str]
    suspicious_frames: Optional[list]
    video_metadata:   Optional[dict]
    processing_time_s: Optional[float]
    error:            Optional[str]
    created_at:       str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": settings.VERSION,
        "device": settings.DEVICE,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file.  Returns a job_id used to trigger analysis.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. "
                   f"Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 ** 2)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB (max {settings.MAX_UPLOAD_MB} MB)",
        )

    # Save file
    job_id   = str(uuid.uuid4())
    save_dir = settings.UPLOAD_DIR / job_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename

    with open(save_path, "wb") as f:
        f.write(content)

    _jobs[job_id] = {
        "status":    "pending",
        "file_path": str(save_path),
        "filename":  file.filename,
        "size_mb":   round(size_mb, 2),
        "created_at": datetime.utcnow().isoformat(),
        "result":    None,
    }

    logger.info(f"Upload: {file.filename} ({size_mb:.1f} MB) → job_id={job_id}")
    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        size_mb=round(size_mb, 2),
        message="Upload successful. Call POST /api/analyze/{job_id} to start analysis.",
    )


@app.post("/api/analyze/{job_id}", response_model=AnalysisResponse)
async def analyze(job_id: str, background_tasks: BackgroundTasks):
    """
    Trigger analysis for an uploaded video.
    Analysis runs in a background task; poll /api/result/{job_id}.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _jobs[job_id]
    if job["status"] in ("processing", "done"):
        return _job_to_response(job_id, job)

    _jobs[job_id]["status"] = "processing"
    background_tasks.add_task(_run_analysis, job_id)

    return _job_to_response(job_id, _jobs[job_id])


@app.get("/api/result/{job_id}", response_model=AnalysisResponse)
async def get_result(job_id: str):
    """
    Poll analysis result.  Status: pending | processing | done | error.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _job_to_response(job_id, _jobs[job_id])


@app.post("/api/analyze_sync", response_model=AnalysisResponse)
async def analyze_sync(file: UploadFile = File(...)):
    """
    Convenience endpoint: upload + analyze in one call (blocks until done).
    Best for small files or development.
    """
    # Reuse upload logic
    upload_resp = await upload_video(file)
    job_id = upload_resp.job_id

    # Run synchronously
    _jobs[job_id]["status"] = "processing"
    await _run_analysis_async(job_id)
    return _job_to_response(job_id, _jobs[job_id])


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Clean up uploaded files and job record."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Remove files
    job_dir = settings.UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    del _jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs (without result payloads to keep response small)."""
    return [
        {
            "job_id":     jid,
            "status":     j["status"],
            "filename":   j.get("filename"),
            "created_at": j.get("created_at"),
        }
        for jid, j in _jobs.items()
    ]


# ─── Background task ──────────────────────────────────────────────────────────

async def _run_analysis_async(job_id: str):
    """Async wrapper around the (blocking) inference pipeline."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_analysis_sync, job_id)


def _run_analysis(job_id: str):
    """FastAPI BackgroundTask entrypoint."""
    _run_analysis_sync(job_id)


def _run_analysis_sync(job_id: str):
    """Run the full inference pipeline and update the job record."""
    job = _jobs.get(job_id)
    if not job:
        return

    try:
        pipeline = get_pipeline()
        result: AnalysisResult = pipeline.run(job["file_path"])
        _jobs[job_id]["status"] = "done" if not result.error else "error"
        _jobs[job_id]["result"] = result
        logger.info(f"Job {job_id} complete → {result.label}")
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["result"] = AnalysisResult(
            label="ERROR", fake_probability=0.5, confidence="LOW", error=str(e)
        )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _job_to_response(job_id: str, job: dict) -> AnalysisResponse:
    r: Optional[AnalysisResult] = job.get("result")
    return AnalysisResponse(
        job_id=job_id,
        status=job["status"],
        label=r.label if r else None,
        fake_probability=r.fake_probability if r else None,
        confidence=r.confidence if r else None,
        video_score=r.video_score if r else None,
        audio_score=r.audio_score if r else None,
        text_score=r.text_score if r else None,
        frame_analysis=r.frame_analysis if r else None,
        modality_details=r.modality_details if r else None,
        explanation=r.explanation if r else None,
        audio_plot=r.audio_plot if r else None,
        breakdown_plot=r.breakdown_plot if r else None,
        suspicious_frames=r.suspicious_frames if r else None,
        video_metadata=r.video_metadata if r else None,
        processing_time_s=r.processing_time_s if r else None,
        error=r.error if r else None,
        created_at=job.get("created_at", ""),
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,       # keep 1 worker so pipeline is shared
    )
