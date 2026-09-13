"""
FastAPI backend — entrypoint.

POST /analyze accepts a video file, runs it through the detection+tracking
pipeline and every registered analyzer, and returns the aggregated
predicted segments as JSON.

Run with: uvicorn app.main:app --reload --port 8000
"""

import json
import os
import shutil
import uuid
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline.aggregator import segments_to_dict
from app.pipeline.analyzer_runner import (
    TRACK_BASED_ACTIVITIES,
    run_selected_analyzers,
    validate_activities,
)
from app.pipeline.detect_track import run_detection_tracking

app = FastAPI(title="Activity Detection & Classification API")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Default full-frame zone
DEFAULT_ZONE = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
LOITERING_DWELL_THRESHOLD_SEC = 10.0


@app.get("/")
def root():
    return {"status": "ok", "message": "Activity Detection API is running"}


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    zone: Optional[str] = Form(None),
    activities: Optional[str] = Form(None),
):
    video_id = str(uuid.uuid4())
    saved_filename = f"{video_id}_{file.filename}"
    video_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse custom zone polygon if provided (e.g. from frontend UI), otherwise fallback to DEFAULT_ZONE
    zone_polygon: List[Tuple[float, float]] = DEFAULT_ZONE
    if zone:
        try:
            parsed = json.loads(zone)
            if isinstance(parsed, list) and len(parsed) >= 3:
                zone_polygon = [(float(pt[0]), float(pt[1])) for pt in parsed]
        except Exception:
            zone_polygon = DEFAULT_ZONE

    selected_activities = None
    if activities:
        try:
            parsed_activities = json.loads(activities)
            if not isinstance(parsed_activities, list) or not all(
                isinstance(activity, str) for activity in parsed_activities
            ):
                raise ValueError("activities must be a JSON list of strings")
            selected_activities = validate_activities(parsed_activities)
        except (ValueError, json.JSONDecodeError):
            return JSONResponse(
                status_code=422,
                content={"detail": "activities must be a valid JSON list of supported activity names"},
            )

    activities_to_run = validate_activities(selected_activities)
    frames = (
        run_detection_tracking(video_path)
        if activities_to_run & TRACK_BASED_ACTIVITIES
        else []
    )
    all_segments = run_selected_analyzers(
        frames,
        video_path,
        zone_polygon,
        activities=selected_activities,
        loitering_dwell_threshold_sec=LOITERING_DWELL_THRESHOLD_SEC,
    )

    return JSONResponse(
        {
            "video_id": video_id,
            "filename": file.filename,
            "video_url": f"/uploads/{saved_filename}",
            "zone_used": zone_polygon,
            "activities_run": sorted(activities_to_run),
            "segments": segments_to_dict(all_segments),
        }
    )


# Static mounts: uploads for video streaming, outputs for charts/results, static for web interface
app.mount("/uploads", StaticFiles(directory="data/uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="app/frontend", html=True), name="static")
