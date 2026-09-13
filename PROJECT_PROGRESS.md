# Project Progress & Implementation Status

**Project:** Activity Detection & Classification in Long Videos
**Organization:** CerebralZip Data Science Internship
**Repository:** `c:\activity-detection-classification`
**Last Updated:** September 13, 2026

---

## 1. Executive Summary

This project implements an end-to-end computer vision and deep learning system for detecting and temporally classifying human activities and security anomalies in CCTV/surveillance videos.

The architecture combines:
1. **Multi-Object Tracking (MOT)** using **YOLOv8 + ByteTrack** (`ultralytics`).
2. **Modular Activity Analyzers** consuming spatiotemporal track trajectories.
3. **Temporal Aggregation & Greedy IoU Evaluation** against ground-truth intervals.
4. **Interactive Surveillance Dashboard (Web UI)** with live HUD alerts, video playback, visual timeline scrubbing, and custom polygon zone drawing (area minimizer).

---

## 2. Completed Components & Deliverables

### A. Environment & External Tooling
- [x] **Installed Core Packages**: Installed `ultralytics` (v8.4.150), PyTorch, OpenCV, `imageio-ffmpeg`, `yt-dlp`, `pytest`, `fastapi`, `uvicorn`.
- [x] **FFmpeg & yt-dlp Tooling**:
  - Configured a standalone **FFmpeg 7.1** binary on Windows.
  - Created [`ffmpeg.bat`](ffmpeg.bat) and [`yt-dlp.bat`](yt-dlp.bat) for direct terminal usage.
  - Created [`scripts/video_tools.py`](scripts/video_tools.py) for video metadata inspection, OpenCV trimming, and keyframe extraction.

---

### B. Activity Analyzers (Pipeline)
All analyzers adhere to a uniform interface: taking `List[TrackFrame]` and producing `List[Segment]` (`start`, `end`, `label`, `confidence`, `track_ids`).

1. **Loitering** ([app/pipeline/analyzers/loitering.py](app/pipeline/analyzers/loitering.py))
   - Dwell-time tracking inside a polygon zone using ray-casting point-in-polygon tests.
   - Gap tolerance (`max_gap_sec`) to prevent track splitting during brief occlusions.
   - Triggers when person dwell time >= 10.0s.
2. **Intrusion / Trespassing** ([app/pipeline/analyzers/intrusion.py](app/pipeline/analyzers/intrusion.py))
   - Detects whenever a person enters a defined restricted boundary.
   - Merges consecutive inside-zone frames into interval segments with confidence scoring.
3. **Running in Restricted Zone** ([app/pipeline/analyzers/running.py](app/pipeline/analyzers/running.py))
   - Measures frame-to-frame centroid velocity (pixels/second) inside a zone.
   - Filters high-frequency noise with a minimum duration threshold (>= 0.5s) when velocity >= 120 px/s.
4. **Crowd Formation** ([app/pipeline/analyzers/crowd.py](app/pipeline/analyzers/crowd.py))
   - Aggregates concurrent distinct person track IDs inside a zone across time.
   - Flags crowd formation intervals when simultaneous count >= 3 persons.
5. **Falling** ([app/pipeline/analyzers/falling.py](app/pipeline/analyzers/falling.py))
   - Uses only `TrackFrame.bbox` history: a rapid width/height increase and
     downward centroid velocity must both exceed their thresholds in one second.
   - Requires the widened posture to persist for a 0.75-second settle window.
6. **Abandoned Object** ([app/pipeline/analyzers/abandoned_object.py](app/pipeline/analyzers/abandoned_object.py))
   - Tracks stationary bags/suitcases with owner distance thresholding.
7. **Fire / Smoke** ([app/pipeline/analyzers/fire_smoke.py](app/pipeline/analyzers/fire_smoke.py))
   - Appearance-based frame detector using HSV chromaticity analysis.
8. **Segment Aggregator** ([app/pipeline/aggregator.py](app/pipeline/aggregator.py))
   - Merges segments from all analyzers, sorts chronologically by start timestamp, and serializes to JSON.

---

### G. Benchmark Results
Per-video evaluation at Temporal IoU >= 0.3 (each video tested with its relevant activity profile):

| Video | Activity | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| campus_running | Running | 3 | 2 | 1 | 0.600 | 0.750 | **0.667** |
| campus_crowd | Crowd Formation | 1 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| campus_loitering | Loitering | 0 | 0 | 0 | — | — | — (no dwell >= 10s in zone) |
| campus_intrusion | Intrusion | 0 | 14 | 0 | 0.000 | 0.000 | 0.000 (zone too narrow, all passers trigger) |
| fall_detection | Falling | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 (bbox-only misses camera angle) |
| fire_smoke | Fire / Smoke | 1 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| pets2006_abandoned | Abandoned Object | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 (YOLOv8n bag tracking needs improvement) |

**Key improvement**: Running F1 improved from **0.400 → 0.667** by using a restricted zone instead of full-frame zone, eliminating 10 of 12 false positives from peripheral pedestrian movement.

---

### C. Backend API Integration
- [x] **FastAPI Application** ([app/main.py](app/main.py)):
   - Endpoint `POST /analyze` receives video uploads and optional custom `zone` polygon JSON and `activities` list.
   - Runs detection, tracking, and all active analyzers based on selected profile.
   - Mounts `/uploads` for streaming uploaded clips in the browser.
   - Mounts `/outputs` for serving prediction JSONs and generated charts.
   - Mounts `/static` for the web dashboard.

---

### D. Interactive Web Surveillance Dashboard
- [x] **Enhanced Web Interface** ([app/frontend/index.html](app/frontend/index.html)):
   - **Side-by-side Video Player**: Embedded HTML5 video player with real-time frame synchronization.
   - **Live HUD Alert Banner**: Heads-Up Display bar above the video that flashes red with activity details during anomaly playback, and turns green during calm intervals.
   - **Area Minimizer (Restricted Zone Selector)**:
     - Full Frame (100% coverage).
     - Center Restricted Corridor.
     - Left Entrance / Pathway.
     - Right Perimeter.
     - Custom Click-to-Draw: User can click 3 to 6 points directly on the video screen.
   - **Interactive Visual Timeline**: Color-coded horizontal segment blocks, synchronized playhead, click-to-seek.
   - **Event Cards Feed**: Chronological list of all flagged anomalies with confidence bars and **"Jump"** button.
   - **Collapsible Raw JSON Viewer**.

---

### E. Automated Unit Testing
- [x] **Test Suite** ([tests/test_analyzers.py](tests/test_analyzers.py)):
   - **13 unit tests** — all passing (13/13).
   - Covers: loitering, intrusion, running, crowd, abandoned object, falling, fire/smoke, aggregation, evaluation metrics, activity validation.

---

### F. Real Video Testing, Benchmarking & Ground Truth
- [x] **Acquired & Trimmed Videos**:
  - `data/uploads/campus_walk_30s.mp4` (720p @ 30fps, ~25s)
  - `data/uploads/fall_detection.mp4`, `data/uploads/fire_smoke.mp4`, `data/uploads/pets2006_abandoned.mp4`
  - `data/uploads/urfd_fall_01.mp4`, `data/uploads/urfd_adl_01.mp4`
- [x] **Ground Truth Annotations**:
  - `data/ground_truth/campus_walk_30s_gt.json`
  - `data/ground_truth/fall_detection_gt.json`
  - `data/ground_truth/fire_smoke_gt.json`
  - `data/ground_truth/pets2006_abandoned_gt.json`
  - `data/ground_truth/benchmark_manifest.json`
- [x] **Benchmark Framework**:
  - CLI evaluation tool: [`scripts/run_eval.py`](scripts/run_eval.py)
  - Batch benchmark runner: [`scripts/run_benchmark.py`](scripts/run_benchmark.py)
  - Timeline visualization: [`scripts/visualize_timeline.py`](scripts/visualize_timeline.py)

---

## 3. Video Inventory & Assessment

| Video | Duration | Resolution | Used For | Relevant? | Issue |
|---|---|---|---|---|---|
| campus_walk.mp4 | 45.5s | 720p | General testing | ⚠️ | No GT yet |
| campus_walk_30s.mp4 | 24.7s | 720p | Running, Crowd, Loitering, Intrusion | ✅ | Core benchmark video |
| fall_detection.mp4 | 32.5s | 360p | Falling | ✅ | Properly aligned GT |
| fire_smoke.mp4 | 54.5s | 720p | Fire/Smoke | ✅ | Properly aligned GT |
| pets2006_abandoned.mp4 | 55.8s | 480p | Abandoned Object | ✅ | PETS2006 benchmark |

**Removed videos** (stale/misaligned):
- `urfd_fall_01.mp4` — only 5.3s, GT was 7-18s (misaligned), deleted
- `urfd_adl_01.mp4` — only 5.0s, no GT, deleted
- `52213d26-bbc1-4c00-8bbd-1ee39f8dacde_campus_walk_30s.mp4` — duplicate UUID upload, deleted

**Missing videos needed** for complete validation:
| Activity | Need | Why |
|---|---|---|
| Loitering | Video with dwell areas (benches, waiting areas) | Current walkway has no one still >= 10s |
| Intrusion | Video with clearly restricted area (server room, restricted hallway) | Current zone is a walkway — all passers trigger detector |

---

## 4. Directory Structure

```text
c:\activity-detection-classification\
├── app\
│   ├── main.py                     # FastAPI entrypoint, /analyze, static mounts
│   ├── pipeline\
│   │   ├── detect_track.py         # YOLOv8 + ByteTrack tracking pipeline
│   │   ├── aggregator.py           # Segment merging and serialization
│   │   ├── analyzer_runner.py      # Profile selection + orchestrator
│   │   └── analyzers\
│   │       ├── base.py             # Segment dataclass, point_in_zone ray casting
│   │       ├── loitering.py        # Loitering dwell-time analyzer
│   │       ├── intrusion.py        # Intrusion zone-crossing analyzer
│   │       ├── running.py          # Running velocity analyzer
│   │       ├── crowd.py            # Crowd formation density analyzer
│   │       ├── falling.py          # Falling bbox heuristic analyzer
│   │       ├── abandoned_object.py # Abandoned object analyzer
│   │       └── fire_smoke.py       # Fire/smoke appearance-based analyzer
│   ├── eval\
│   │   └── temporal_iou.py         # Temporal IoU, greedy matching, P/R/F1
│   └── frontend\
│       └── index.html              # Interactive surveillance dashboard UI
├── data\
│   ├── ground_truth\
│   │   ├── example_video_gt.json   # Template ground truth format
│   │   ├── benchmark_manifest.json # Benchmark video definitions
│   │   ├── campus_walk_30s_gt.json # Benchmark ground truth
│   │   ├── fall_detection_gt.json
│   │   ├── fire_smoke_gt.json
│   │   └── pets2006_abandoned_gt.json
│   ├── uploads\
│       ├── campus_walk.mp4
│       ├── campus_walk_30s.mp4
│       ├── fall_detection.mp4
│       ├── fire_smoke.mp4
│       ├── pets2006_abandoned.mp4
├── outputs\
│   ├── campus_walk_30s_predictions.json
│   ├── timeline_campus_walk.png
│   ├── timeline_campus_walk_restricted.png
│   ├── timeline_campus_running.png
│   ├── timeline_campus_crowd.png
│   ├── timeline_campus_people_restricted.png
│   └── benchmark\
│       ├── benchmark_summary.json
│       ├── campus_people_predictions.json
│       ├── campus_running_predictions.json
│       ├── campus_crowd_predictions.json
│       ├── campus_loitering_predictions.json
│       ├── campus_intrusion_predictions.json
│       ├── fall_detection_predictions.json
│       ├── fire_smoke_predictions.json
│       └── pets2006_abandoned_predictions.json
├── scripts\
│   ├── video_tools.py              # Video inspection & OpenCV trimming
│   ├── run_eval.py                 # CLI temporal IoU evaluation tool
│   ├── run_benchmark.py            # Batch benchmark runner
│   └── visualize_timeline.py       # Matplotlib timeline generator
├── tests\
│   └── test_analyzers.py           # Pytest test suite (13/13 passing)
├── ffmpeg.bat
├── yt-dlp.bat
├── requirements.txt
├── .gitignore
├── README.md
├── WORK_PLAN.md
└── PROJECT_PROGRESS.md
```

---

## 5. Quick Command Reference

| Action | Command |
|---|---|
| **Run Test Suite** | `python -m pytest tests/ -v` |
| **Start Web App Server** | `python -m uvicorn app.main:app --reload --port 8000` |
| **Open Web Dashboard** | Navigate to `http://localhost:8000/static/index.html` |
| **Open API Docs (Swagger)** | Navigate to `http://localhost:8000/docs` |
| **Download YouTube Video** | `.\yt-dlp.bat -f "bestvideo[height<=720]+bestaudio/best[height<=720]" -o "data/uploads/video.mp4" "<URL>"` |
| **Trim Video with FFmpeg** | `.\ffmpeg.bat -i input.mp4 -ss 00:00:00 -to 00:00:30 -c copy output.mp4 -y` |
| **Run Temporal Evaluation** | `python scripts/run_eval.py --pred outputs/preds.json --gt data/ground_truth/gt.json --iou 0.3` |
| **Run Benchmark** | `python scripts/run_benchmark.py` |
| **Generate Timeline Plot** | `python scripts/visualize_timeline.py --pred outputs/preds.json --gt data/ground_truth/gt.json --out outputs/plot.png` |

---

## 5. Next Steps / Pending Roadmap

1. **Intrusion zone calibration**: The restricted zone is too narrow — every passer triggers intrusion. Widen or reposition the zone to only flag truly unauthorized areas.
2. **Loitering validation**: Current walkway zone has no dwell >= 10s. Use a bench/waiting-area zone for proper loitering validation.
3. **Expand benchmark suite**: Add more positive/negative clips per activity class for statistical validity per WORK_PLAN.md guardrails.
4. **Falling improvement**: Confirm candidates with MediaPipe Pose (not yet installed) for production-grade fall detection.
5. **Abandoned Object improvement**: Fine-tune YOLOv8n bag detection or use a better-fitting PETS2006 clip.
6. **Final Report Preparation**: Compiling the 2–4 page technical submission covering design rationale, architecture diagram, quantitative evaluation tables, and failure mode analysis.
