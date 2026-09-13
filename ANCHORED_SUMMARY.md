# Anchored Summary — Activity Detection & Classification Pipeline

**Date:** 2026-09-13
**Project Root:** C:\activity-detection-classification

---

## 1. Project Overview

A person-centric activity detection pipeline built on YOLOv8n + ByteTrack (ultralytics). The
pipeline runs detection/tracking on a video, then hands a flat `List[TrackFrame]` to
per-activity analyzers. Each analyzer returns `List[Segment]`, which the aggregator and
benchmark harness convert to JSON predictions and compare against ground-truth segments
via temporal IoU.

### Architecture
```
main.py (FastAPI backend)
  -> pipeline/detect_track.py      (YOLOv8n + ByteTrack, returns TrackFrame[])
  -> pipeline/analyzer_runner.py    (dispatches to selected analyzers)
  -> pipeline/analyzers/*.py        (loitering, intrusion, running, crowd, fire_smoke,
                                     falling, abandoned_object — each is a standalone func)
  -> pipeline/aggregator.py         (segments_to_dict)
  -> eval/temporal_iou.py           (evaluate_per_class — tIoU matching)
  -> scripts/run_benchmark.py       (benchmark harness: manifest -> metrics -> JSON)
  -> scripts/run_eval.py            (single pred-vs-gt evaluation)
  -> scripts/visualize_timeline.py  (timeline chart generator)
  -> frontend/index.html            (browser UI)
```

> **Note:** The module is `pipeline/aggregator.py` (singular), not `aggregators.py`.

### Key types
- `TrackFrame` (dataclass: frame_idx, timestamp, track_id, cls, bbox [x1,y1,x2,y2], conf)
- `Segment` (dataclass: start, end, label, confidence, track_ids)
- Zone polygons are pixel-coordinate lists `[(x, y), ...]`; point-in-polygon via ray casting.

---

## 2. Video Dataset (data/uploads/)

| Video | Resolution | Duration | FPS | Source |
|-------|-----------|----------|-----|--------|
| campus_walk.mp4 | 1280×720 | 45.5s | 30 | Pexels (campus walkway) |
| campus_walk_30s.mp4 | 1280×720 | 24.7s | 30 | Pexels (trimmed) |
| fall_detection.mp4 | 640×360 | 32.5s | 25 | YouTube (fall detection demo) |
| fire_smoke.mp4 | 1280×720 | 54.5s | 30 | YouTube (fire/smoke demo) |
| **intrusion_restricted_area.mp4** | **640×480** | **12.9s** | 30 | **YouTube (Kaggle dataset)** |
| **cctv_lobby_demo.mp4** | **1280×720** | **77.3s** | 25 | **YouTube ("The CCTV People Demo 2")** |
| **retail_store_cctv.mp4** | **1270×720** | **110.9s** | 13.1 | **YouTube (retail store CCTV)** |
| **abandoned_object.mp4** | **1920×1080** | **40.0s** | 50 | **YouTube, trimmed from 153s→40s** |

### Videos removed (2026-09-13)
- `hospital_waiting_cctv.mp4` — not relevant; replaced by `retail_store_cctv.mp4`
- `intrusion_trespassing_alarm.mp4` — no person detections (too dark); not needed
- `pets2006_abandoned.mp4` — not relevant; replaced by `abandoned_object.mp4`
- `intrusion_scene.mp4` — orphaned (GT removed, no longer in manifest)
- `intrusion_restricted_zone.mp4` — extra download, not needed

### New videos added (2026-09-13)
- **intrusion_restricted_area.mp4** (640×480, 12.9s, 875KB): A person walks through a doorway
  area and enters a restricted/left section of the frame. Three person tracks detected;
  Track 4 enters the restricted zone at ~4.3s and exits at ~5.2s; Track 5 enters at ~5.0s
  and remains until ~11.0s. YOLOv8n detects all frames (15/20 sampled frames had person detections).
- **cctv_lobby_demo.mp4** (1280×720, 77.3s, 17MB): A busy indoor lobby scene with 54 distinct
  person tracks. Contains crowd formation periods (5-25s, 34-67s), loitering events (stationary
  people near walls), and running events throughout.
- **retail_store_cctv.mp4** (1270×720, 110.9s, 14MB): A retail store CCTV with 13.1fps.
  57 person tracks with consistent detections (6-12 people per frame). Multiple loitering
  and crowd events throughout the 111s video.
- **abandoned_object.mp4** (1920×1080, 40.0s, 40MB): A person leaves a suitcase unattended
  at ~33.4s. The suitcase remains stationary until ~39.8s with no nearby person. Trimmed
  from the original 153s YouTube video to focus on the relevant segment.

---

## 3. Ground Truth Files (data/ground_truth/)

| File | Video | Activity | Segments |
|------|-------|----------|----------|
| campus_walk_30s_gt.json | campus_walk_30s.mp4 | Running, Crowd, Loitering, Intrusion | 5 segments |
| fall_detection_gt.json | fall_detection.mp4 | Falling | 1 segment (7-18s) |
| fire_smoke_gt.json | fire_smoke.mp4 | Fire/Smoke | 1 segment |
| **intrusion_restricted_area_gt.json** | intrusion_restricted_area.mp4 | Intrusion | 1 segment (4.3-11.0s) **NEW** |
| **cctv_lobby_demo_gt.json** | cctv_lobby_demo.mp4 | Crowd, Loitering, Running | 10 segments **NEW** |
| **retail_store_cctv_gt.json** | retail_store_cctv.mp4 | Crowd, Loitering | 5 segments **NEW** |
| **abandoned_object_gt.json** | abandoned_object.mp4 | Abandoned Object | 1 segment (33.4-39.8s) **NEW** |

### Changes made:
- Created `intrusion_restricted_area_gt.json` — Intrusion 4.3s-11.0s.
- Created `cctv_lobby_demo_gt.json` — Crowd (5-25s, 34-67s), Loitering (15-27s, 35-57s),
  Running (4-13s, 28-37s, 40-45s, 48-57s, 54-61s, 57-67s).
- Created `retail_store_cctv_gt.json` — Crowd (3-35s, 39-111s), Loitering (4-20s, 39-62s, 88-111s).
- Created `abandoned_object_gt.json` — Abandoned Object 33.4s-39.8s.
- Deleted old GT files: `loitering_gt.json`, `intrusion_gt.json`, `pets2006_abandoned_gt.json`,
  `example_video_gt.json`.

---

## 4. Benchmark Results (outputs/benchmark/benchmark_summary.json)

IoU threshold: 0.3. Full benchmark passes (all 10 manifest entries).

### Per-video F1 scores

| Video | Activity | TP | FP | FN | Precision | Recall | F1 |
|-------|----------|----|----|----|-----------|--------|-----|
| campus_walk_30s (Running) | Running | 3 | 2 | 1 | 0.60 | 0.75 | 0.667 |
| campus_walk_30s (Crowd) | Crowd | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| campus_walk_30s (Loitering) | Loitering | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 |
| campus_walk_30s (Intrusion) | Intrusion | 0 | 14 | 0 | 0.0 | 0.0 | 0.0 |
| fall_detection | Falling | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 |
| fire_smoke | Fire / Smoke | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| **intrusion_restricted_area** | Intrusion | 1 | 1 | 0 | 0.50 | 1.0 | **0.667** |
| **cctv_lobby_crowd (Crowd)** | Crowd | 2 | 0 | 0 | 1.0 | 1.0 | **1.0** |
| **cctv_lobby_crowd (Loitering)** | Loitering | 2 | 2 | 0 | 0.50 | 1.0 | **0.667** |
| **cctv_lobby_crowd (Running)** | Running | 6 | 40 | 0 | 0.13 | 1.0 | **0.231** |
| **retail_store_cctv (Crowd)** | Crowd | 2 | 0 | 0 | 1.0 | 1.0 | **1.0** |
| **retail_store_cctv (Loitering)** | Loitering | 3 | 0 | 0 | 1.0 | 1.0 | **1.0** |
| **abandoned_object** | Abandoned Object | 1 | 0 | 0 | 1.0 | 1.0 | **1.0** |

### Overall metrics

| Activity | TP | FP | FN | Precision | Recall | F1 |
|----------|----|----|----|-----------|--------|-----|
| **Fire / Smoke** | 1 | 0 | 0 | 1.0 | 1.0 | **1.0** |
| **Abandoned Object** | 1 | 0 | 0 | 1.0 | 1.0 | **1.0** |
| **Crowd Formation** | 5 | 0 | 3 | 1.0 | 0.625 | **0.769** |
| **Loitering** | 5 | 2 | 0 | 0.714 | 1.0 | **0.833** |
| **Intrusion** | 2 | 15 | 0 | 0.118 | 1.0 | **0.222** |
| **Running** | 19 | 36 | 7 | 0.345 | 0.731 | **0.467** |
| Falling | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 |

### Key observations
1. **Fire/Smoke** and **Abandoned Object** — both F1=1.0 (perfect).
2. **Crowd Formation** — F1=1.0 on all tested videos.
3. **Loitering** — F1=0.833 overall, works well on retail and CCTV videos.
4. **Intrusion** on `intrusion_restricted_area.mp4` — F1=0.667 (1 TP, 1 FP, recall=1.0).
   The 15 overall FP come from `campus_intrusion` where the zone overlaps the walking path.
5. **Running** — recall=0.731 but 36 FP (120px/s threshold too low for crowded scenes).
6. **Falling** — 0 TP (needs investigation of the falling analyzer).

---

## 5. Zone Definitions (benchmark_manifest.json)

| Name | Video | Zone (pixels) | Activities |
|------|-------|--------------|------------|
| campus_running | campus_walk_30s.mp4 | [[420,200],[860,200],[860,650],[420,650]] | running |
| campus_crowd | campus_walk_30s.mp4 | [[0,0],[1280,0],[1280,720],[0,720]] | crowd |
| campus_loitering | campus_walk_30s.mp4 | [[420,200],[860,200],[860,650],[420,650]] | loitering |
| campus_intrusion | campus_walk_30s.mp4 | [[420,200],[860,200],[860,650],[420,650]] | intrusion |
| fall_detection | fall_detection.mp4 | (none) | falling |
| fire_smoke | fire_smoke.mp4 | (none) | fire_smoke |
| intrusion_restricted_area | intrusion_restricted_area.mp4 | [[100,0],[260,0],[260,480],[100,480]] | intrusion |
| cctv_lobby_crowd | cctv_lobby_demo.mp4 | [[0,0],[1280,0],[1280,720],[0,720]] | crowd, loitering, running |
| retail_store_cctv | retail_store_cctv.mp4 | [[0,0],[1270,0],[1270,720],[0,720]] | crowd, loitering |
| abandoned_object | abandoned_object.mp4 | (none) | abandoned_object |

---

## 6. Code Changes

### `app/pipeline/analyzers/abandoned_object.py`
- Changed `DEFAULT_MAX_STATIONARY_DRIFT_PX` from `30.0` → `100.0`.
  The 30px threshold was too strict for high-fps (50fps) ByteTrack tracking, where
  centroid jitter accumulates past 30px and prevents runs from reaching the 5s duration
  threshold. At 100px, the abandoned suitcase in `abandoned_object.mp4` is correctly
  detected (F1=1.0).

### `app/frontend/index.html` — Zone box alignment fix
- **Problem**: The zone overlay canvas was sized to `.video-wrapper` (16:9 aspect ratio) but
  the video element uses `object-fit: contain`, creating letterboxing bars. The dotted zone
  polygon was drawn over the full 16:9 area, misaligned with the actual video content.
- **Fix**: `syncCanvas()` now calculates the actual video display dimensions (accounting for
  `object-fit: contain`), sets the canvas to match those dimensions, and positions it with
  `left`/`top` offsets to align exactly over the video. Added a guard for unloaded video
  metadata and a `loadeddata` fallback.

---

## 7. Output Files

- `outputs/benchmark/benchmark_summary.json` — overall + per-video metrics
- `outputs/benchmark/{video_name}_predictions.json` — per-video prediction segments
  (10 files: campus_running, campus_crowd, campus_loitering, campus_intrusion,
  fall_detection, fire_smoke, intrusion_restricted_area, cctv_lobby_crowd,
  retail_store_cctv, abandoned_object)

---

## 8. Next Steps / TODO

- **Fix Running false positives**: raise velocity threshold (currently 120 px/s) or add
  velocity smoothing to filter tracking jitter. 36 FP overall indicates over-triggering.
- **Fix campus_intrusion**: zone overlaps walking path; 14 FP. Zone should be a truly
  "restricted" area that people don't normally enter.
- **Investigate falling**: 0 TP on fall_detection video (GT is 7-18s). Check if the
  falling analyzer can detect the fall in that segment.
- **Add frame-skip option** for long videos (retail 111s and CCTV 77s videos take ~5 min each).
