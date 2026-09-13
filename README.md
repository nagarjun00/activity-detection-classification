# Activity Detection & Classification in Long Videos

## Submission Requirements ✅

- ✅ Clearly defined set of 7 activity classes (out of 10)
- ✅ System outputs predicted activity segments with start time, end time, class label, and confidence score
- ✅ Quantitative evaluation (precision, recall, F1 at IoU >= 0.3) with metric justification
- ✅ Qualitative analysis: visualizations of predictions vs. ground truth, plus failure mode analysis

See [`REPORT.md`](REPORT.md) for the full 2-4 page report.

CerebralZip Data Science Internship — assignment submission.

## Scope decision (read this first)

Of the 10 listed activity classes, this system targets **7**, chosen by data
availability and time budget, not difficulty alone:

| Class | Approach | Status |
|---|---|---|
| Loitering | Zone + dwell time on person tracks | **Implemented** |
| Intrusion / Trespassing | Zone-crossing on person tracks | **Implemented** |
| Running (restricted zone) | Track centroid velocity threshold | **Implemented** |
| Crowd Formation | Person count/density over time in a region | **Implemented** |
| Abandoned Object | Object track stationary + separated from any person | **Implemented** |
| Fire / Smoke | Appearance-based frame detector | **Implemented** |
| Falling | Bbox aspect-ratio flip + downward velocity + settle check | **Implemented** |

**Explicitly out of scope:** Fighting/Physical Assault, Vandalism,
Theft/Snatching. These need learned action-recognition models (appearance
and motion patterns, not just detection+tracking rules), and no usable
labeled dataset with precise temporal boundaries exists for them within a
48-hour, self-annotated-data budget. A production version of this system
would fine-tune a Kinetics-pretrained action recognition backbone (e.g.
X3D or SlowFast) on labeled clips for these three. State this reasoning
in the report — it's a deliberate judgment call, not an oversight.

## Architecture

```
Video input (YouTube clips or local files)
        v
Frame extraction (OpenCV via ultralytics)
        v
Detection & tracking (YOLOv8n + ByteTrack)   <- reused from prior ANPR/traffic project
        v
Per-class analyzers (7 modules, one per activity class)
        v
Segment aggregation (merge into intervals + confidence)
        v
Output (JSON) + Evaluation (temporal IoU, precision/recall)
```

The system supports **video activity profiles** — select which analyzers
run per video to avoid unrelated false alerts (e.g. don't run fire/smoke
detection on a walkway clip). Profiles are available in the web
dashboard and via the API `activities` parameter.

Backend: FastAPI wraps the whole pipeline behind a single `/analyze`
endpoint. Frontend is a modern surveillance dashboard with video
playback, zone overlay, interactive timeline, event feed, and raw JSON
inspector.

```
app/
├── main.py                        # FastAPI entrypoint, /analyze endpoint, static mounts
├── pipeline/
│   ├── detect_track.py            # YOLOv8 + ByteTrack, returns List[TrackFrame]
│   ├── aggregator.py              # merges analyzer outputs into final segments
│   ├── analyzer_runner.py         # profile selection + orchestrator
│   └── analyzers/
│       ├── base.py                # shared Segment type, point_in_zone helper
│       ├── loitering.py           # dwell-time-in-zone rule
│       ├── intrusion.py           # zone-crossing rule
│       ├── running.py             # velocity-in-zone rule
│       ├── crowd.py               # person density rule
│       ├── abandoned_object.py    # stationary object + no-owner rule
│       ├── falling.py             # bbox aspect/velocity/settle heuristic
│       └── fire_smoke.py          # appearance-based chromaticity rule
├── eval/
│   └── temporal_iou.py            # precision/recall/F1 at IoU threshold
└── frontend/
    └── index.html                 # interactive surveillance dashboard
data/
├── uploads/                       # videos land here at runtime
└── ground_truth/
    ├── benchmark_manifest.json    # benchmark video definitions
    └── <name>_gt.json             # ground truth annotations
outputs/
├── benchmark/                     # benchmark predictions and summary
└── timeline*.png                  # timeline visualizations
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000/static/index.html` to upload a video
through the browser, or hit the API directly:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@data/uploads/your_video.mp4" \
  -F "activities=[\"loitering\",\"intrusion\",\"running\"]" \
  -F "zone=[[0,0],[1280,0],[1280,720],[0,720]]"
```

Response shape:

```json
{
  "video_id": "...",
  "filename": "your_video.mp4",
  "segments": [
    { "start": 12.5, "end": 34.0, "label": "Loitering", "confidence": 0.85, "track_ids": [3] }
  ]
}
```

## Evaluation

```python
from app.eval.temporal_iou import evaluate_per_class
import json

predictions = json.load(open("outputs/your_video_predictions.json"))["segments"]
ground_truth = json.load(open("data/ground_truth/example_video_gt.json"))["segments"]

results = evaluate_per_class(predictions, ground_truth, iou_threshold=0.3)
print(results)
```

**Metric choice, for the report:** temporal IoU between predicted and
ground-truth intervals, matched greedily per class, then precision/recall/F1
at IoU >= 0.3. This mirrors the IoU concept used in spatial object
detection (mAP/IoU), applied to time intervals instead of spatial boxes.
0.3 rather than the spatial-detection-standard 0.5 is a deliberate choice:
activity boundaries are fuzzier to hand-annotate than object edges, so
0.5 would penalize reasonable predictions for boundary disagreement rather
than genuine misses.

## Video profiles and repeatable benchmarking

Do not run every heuristic on every video. In the dashboard, select the
profile matching the uploaded clip: people/zone monitoring, fall, abandoned
luggage, fire/smoke, or exploratory all-analyzers mode. This prevents an
unrelated detector (for example, fire/smoke on a campus walkway) from being
reported as an alert.

Run the labelled local benchmark suite with:

```bash
python scripts/run_benchmark.py
```

The manifest at `data/ground_truth/benchmark_manifest.json` defines each
video, its ground truth, its relevant activities, and (where applicable) the
camera zone. The command writes one prediction file per video and
`outputs/benchmark/benchmark_summary.json`. Add several positive and negative
clips from each target camera before changing thresholds; a single video
cannot validate the seven activity classes.

## Zone configuration

Zone-based analyzers (loitering, intrusion, running, crowd) require a
polygon matching the camera's field of view. The default zone covers the
full frame. For best results, use a restricted zone (e.g. a walkway
corridor) for running and intrusion detection to eliminate background
pedestrian triggers. Zones are specified as pixel coordinates in the
API request or selected via the dashboard's Area Minimizer toolbar.

## Status — what's done vs left to build

**Done:**
- Detection + tracking pipeline (`detect_track.py`) — YOLO/ByteTrack
  integration via `ultralytics`
- All 7 activity analyzers — fully implemented and tested
- Aggregator, temporal IoU eval module — both tested and working
- FastAPI backend with `/analyze` endpoint and modern web dashboard
- Interactive frontend with zone overlay, timeline, event feed, profiles
- Automated test suite (13 tests, all passing)
- Benchmark framework with manifest-driven configuration

**Future improvements:**
- MediaPipe Pose confirmation for falling candidates (would require
  retaining frame crops in `detect_track.py`)
- Fine-tuned fire/smoke YOLO checkpoint replacing the chromaticity heuristic
- Additional positive/negative benchmark clips for each activity class
- Abandoned object model fine-tuning for PETS2006 clips

## Datasets

- **Loitering / Intrusion / Running / Crowd Formation:** no pre-existing
  labeled dataset needed — these run on your own tracker output. Pick
  1-3 CCTV-style YouTube clips, hand-timestamp ground truth for each
  (start/end/label), save in `data/ground_truth/` following
  `example_video_gt.json`'s format.
- **Abandoned Object:** PETS2006 or i-LIDS AVSS Abandoned Baggage dataset
  — purpose-built for this exact task, ground truth included.
- **Fire/Smoke:** Roboflow Universe / Kaggle fire-smoke detection sets,
  several already YOLO-annotated.
- **Falling:** UR Fall Detection Dataset or Le2i Fall Detection Dataset.

### Falling limitation and future work

Falling confidence is a heuristic based on threshold exceedances, not a
calibrated probability. A production system should confirm bbox candidates
with MediaPipe Pose on each person's cropped frame. That would require
retaining frame images in `detect_track.py`; the current pipeline streams
and discards them for memory efficiency, which is outside this assignment's
scope.

## Report (2-4 page deliverable)

See [`REPORT.md`](REPORT.md) for the full technical report covering:

- **Scope decision** — why 7 classes are targeted and why 3 are excluded
- **Architecture diagram** — full pipeline description
- **Per-class approach** — each of the 7 activity analyzers
- **Quantitative results** — precision/recall/F1 per class from `evaluate_per_class` at IoU >= 0.3, with metric justification
- **Qualitative analysis** — timeline plots comparing predicted vs. ground-truth segments for sample videos, plus discussion of success modes and failure cases

The report addresses all four submission requirements:
1. Clearly defined activity classes
2. Output format (start/end/label/confidence per segment)
3. Quantitative evaluation with metric justification
4. Qualitative analysis with visualizations
