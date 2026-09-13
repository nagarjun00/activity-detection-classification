# Activity Detection & Classification Report
## CerebralZip Data Science Internship

### 1. Executive Summary

This project implements an end-to-end computer vision system for detecting and temporally classifying human activities and security anomalies in CCTV/surveillance videos. The architecture combines YOLOv8 + ByteTrack multi-object tracking with seven modular activity analyzers, a segment aggregator, and temporal IoU-based evaluation. The system targets 7 of 10 activity classes, deliberately excluding 3 that require learned action-recognition models beyond the assignment's 48-hour self-annotated data budget.

### 2. Scope Decision

Of the 10 listed activity classes, this system targets **7**, chosen by data availability and time budget, not difficulty alone:

| Class | Approach | Status |
|---|---|---|
| Loitering | Zone + dwell time on person tracks | Implemented |
| Intrusion / Trespassing | Zone-crossing on person tracks | Implemented |
| Running (restricted zone) | Track centroid velocity threshold | Implemented |
| Crowd Formation | Person count/density over time in a region | Implemented |
| Abandoned Object | Object track stationary + separated from any person | Implemented |
| Fire / Smoke | Appearance-based frame detector | Implemented |
| Falling | Bbox aspect-ratio flip + downward velocity + settle check | Implemented |

**Explicitly out of scope:** Fighting/Physical Assault, Vandalism, Theft/Snatching. These need learned action-recognition models (appearance and motion patterns, not just detection+tracking rules), and no usable labeled dataset with precise temporal boundaries exists within a 48-hour, self-annotated-data budget. A production version would fine-tune a Kinetics-pretrained action recognition backbone (e.g., X3D or SlowFast) on labeled clips for these three.

### 3. Architecture

```
Video input (YouTube clips or local files)
        v
Frame extraction (OpenCV via ultralytics)
        v
Detection & tracking (YOLOv8n + ByteTrack)
        v
Per-class analyzers (7 modules, one per activity class)
        v
Segment aggregation (merge into intervals + confidence)
        v
Output (JSON) + Evaluation (temporal IoU, precision/recall/F1)
```

**Video profiles** — select which analyzers run per video to avoid unrelated false alerts (e.g., don't run fire/smoke on a walkway clip). Profiles available via API `activities` parameter.

**Backend:** FastAPI wraps the whole pipeline behind a single `/analyze` endpoint. Frontend is a modern surveillance dashboard with video playback, zone overlay, interactive timeline, event feed, and raw JSON inspector.

### 4. Activity Analyzers

All analyzers adhere to a uniform interface: taking `List[TrackFrame]` and producing `List[Segment]` (start, end, label, confidence, track_ids).

1. **Loitering** — Dwell-time tracking inside a polygon zone using ray-casting point-in-polygon tests. Triggers when person dwell time >= 10.0s. Gap tolerance (`max_gap_sec`) prevents track splitting during brief occlusions.

2. **Intrusion / Trespassing** — Detects whenever a person enters a defined restricted boundary. Merges consecutive inside-zone frames into interval segments with confidence scoring.

3. **Running in Restricted Zone** — Measures frame-to-frame centroid velocity (pixels/second) inside a zone. Filters high-frequency noise with a minimum duration threshold (>= 0.5s) when velocity >= 120 px/s. *Key improvement: F1 improved from 0.400 → 0.667 by using a restricted zone instead of full-frame zone, eliminating 10 of 12 false positives.*

4. **Crowd Formation** — Aggregates concurrent distinct person track IDs inside a zone across time. Flags crowd formation intervals when simultaneous count >= 3 persons.

5. **Falling** — Uses only `TrackFrame.bbox` history: a rapid width/height increase and downward centroid velocity must both exceed their thresholds in one second. Requires the widened posture to persist for a 0.75-second settle window. Known limitation: bbox-only approach misses camera angle; MediaPipe Pose confirmation would be needed for production-grade fall detection.

6. **Abandoned Object** — Tracks stationary bags/suitcases with owner distance thresholding. Triggers when object is stationary >= 5s and no person is within 120px.

7. **Fire / Smoke** — Appearance-based frame detector using HSV chromaticity analysis. Detects bright red/orange chromaticity above a threshold.

**Segment Aggregator** — Merges segments from all analyzers, sorts chronologically by start timestamp, and serializes to JSON.

### 5. Evaluation

**Quantitative metrics:** Temporal IoU between predicted and ground-truth intervals, matched greedily per class, then precision/recall/F1 at IoU >= 0.3. This mirrors the IoU concept used in spatial object detection (mAP/IoU), applied to time intervals instead of spatial boxes.

**Metric choice justification:** 0.3 rather than the spatial-detection-standard 0.5 is a deliberate choice. Activity boundaries are fuzzier to hand-annotate than object edges, so 0.5 would penalize reasonable predictions for boundary disagreement rather than genuine misses.

**Benchmark results** (at IoU >= 0.3, per video with relevant activity profile):

| Video | Activity | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| campus_running | Running | 3 | 2 | 1 | 0.600 | 0.750 | 0.667 |
| campus_crowd | Crowd Formation | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| fall_detection | Falling | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 (bbox-only misses camera angle) |
| fire_smoke | Fire / Smoke | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| pets2006_abandoned | Abandoned Object | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 (YOLOv8n bag tracking needs improvement) |

**Key improvement:** Running F1 improved from 0.400 → 0.667 by using a restricted zone instead of full-frame zone.

### 6. Qualitative Analysis

For the report, visualize predictions vs. ground-truth segments on a timeline using `matplotlib` (a horizontal bar per segment, predicted vs. actual, colored by label). Plot 2-3 sample videos and discuss:

- **Successes:** Crowd formation detection working correctly; fire/smoke chromaticity detection reliable; running detection with restricted zone eliminating false positives.
- **Failures/limitations:**
  - Loitering under-detecting due to tracker ID switches when people occlude each other
  - Falling detection missing camera angle (bbox-only heuristic)
  - Intrusion triggering on all passers in a walkway zone (zone too narrow)
  - Abandoned object needing fine-tuned PETS2006 clip or better bag detection
  - No positive/negative benchmark clips for complete class validation

### 7. How to Run

#### Setup

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

#### Running the Web App

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/static/index.html` to upload a video through the browser, or hit the API directly:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@data/uploads/your_video.mp4" \
  -F "activities=[\"loitering\",\"intrusion\",\"running\"]" \
  -F "zone=[[0,0],[1280,0],[1280,720],[0,720]]"
```

**Response shape:**

```json
{
  "video_id": "...",
  "filename": "your_video.mp4",
  "segments": [
    { "start": 12.5, "end": 34.0, "label": "Loitering", "confidence": 0.85, "track_ids": [3] }
  ]
}
```

#### Evaluation

```python
from app.eval.temporal_iou import evaluate_per_class
import json

predictions = json.load(open("outputs/your_video_predictions.json"))["segments"]
ground_truth = json.load(open("data/ground_truth/example_video_gt.json"))["segments"]

results = evaluate_per_class(predictions, ground_truth, iou_threshold=0.3)
print(results)
```

#### Benchmark

```bash
python scripts/run_benchmark.py
```

The manifest at `data/ground_truth/benchmark_manifest.json` defines each video, its ground truth, its relevant activities, and (where applicable) the camera zone. The command writes one prediction file per video and `outputs/benchmark/benchmark_summary.json`.

#### Timeline Visualization

```bash
python scripts/visualize_timeline.py --pred outputs/preds.json --gt data/ground_truth/gt.json --out outputs/plot.png
```

### 8. Directory Structure

```text
c:\activity-detection-classification\
├ app\          # FastAPI + pipeline + analyzers
├ data\         # Uploads + ground truth
├ outputs\      # Predictions + plots + summary
├ scripts\      # Benchmark + eval + visualization tools
├ tests\        # 13 unit tests, all passing
├ requirements.txt
├ REPORT.md     # This document
├ README.md     # Running instructions (updated)
└── yolov8n.pt  # Pretrained YOLOv8n model
```

### 9. Future Improvements

- MediaPipe Pose confirmation for falling candidates (would require retaining frame crops in `detect_track.py`)
- Fine-tuned fire/smoke YOLO checkpoint replacing the chromaticity heuristic
- Additional positive/negative benchmark clips for each activity class
- Abandoned object model fine-tuning for PETS2006 clips
- Final Report Preparation: compiling the 2–4 page technical submission