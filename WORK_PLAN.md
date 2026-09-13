# Focused Validation Plan

**Status:** Active
**Updated:** 2026-09-13
**Goal:** Produce an assignment-quality, evidence-backed demo without trying
to train a production model for every activity class.

## Current architecture

```text
Raw video
  -> YOLOv8n object detection (person, backpack, handbag, suitcase)
  -> ByteTrack track IDs
  -> selected activity analyzer(s)
  -> temporal segments in the frontend and JSON API response
  -> temporal-IoU benchmark evaluation
```

The frontend now has a **Video activity profile** selector. Use it instead of
running all heuristics for every clip:

| Profile | Analyzers run |
|---|---|
| People & restricted-zone monitoring | Loitering, Intrusion, Running, Crowd Formation |
| Fall detection clip | Falling |
| Abandoned luggage clip | Abandoned Object |
| Fire / smoke clip | Fire / Smoke |
| Run every analyzer | Exploratory only; may create unrelated false alerts |

## Baseline benchmark: known facts

Command: `python scripts/run_benchmark.py`

| Class / clip | Current F1 | Interpretation |
|---|---:|---|
| Crowd Formation / campus_crowd | 1.000 | Perfect on 1 labelled clip. |
| Running / campus_running (restricted zone) | 0.667 | Zone-restricted reduced FPs from 12→2; missed 1 GT segment. |
| Falling / fall_detection | 0.000 | Bbox-only rule missed this camera angle. |
| Abandoned Object / pets2006_abandoned | 0.000 | YOLOv8n bag tracking needs improvement. |
| Fire / Smoke / fire_smoke | 1.000 | Coarse temporal metric only. |
| Loitering / campus_walk | 0.000 | No dwell >= 10s in restricted zone. |
| Intrusion / campus_walk | 0.000 | 14 FP — zone too narrow, all passers trigger it. |

These are baseline diagnostics, **not final project scores**.

## Video Inventory & Relevance Assessment

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
- `52213d26-bbc1-4c00-8bbd-1ee39f8dacde_campus_walk_30s.mp4` — duplicate, deleted

**Missing videos needed**:
| Activity | Need | Why |
|---|---|---|
| Loitering | Dedicated video with dwell areas (benches, waiting areas) | Current walkway has no one standing still >= 10s |
| Intrusion | Video with clearly restricted area (server room, restricted hallway) | Current zone is a walkway — all passers trigger it |
| Loitering negative | Normal walkway where people walk through quickly | Needed to verify no false positives |
| Intrusion negative | Normal public area where walking through is allowed | Needed to verify no false positives |

## Target validation suite

Keep clips short (20–60 seconds), static-camera where possible, and add one
normal/negative clip for every risky detector.

| Activity group | Preferred source | Required clips |
|---|---|---|
| Running, Loitering, Intrusion | VIRAT or CUHK Avenue | one positive clip per rule plus one normal walkway clip |
| Crowd Formation | MOT17 or Mall Dataset | one count-increase clip and one low-count clip |
| Falling | UR Fall Detection Dataset | 2–3 fall clips and 2 ADL clips |
| Abandoned Object | PETS 2006 | one leave-luggage clip and one attended-luggage clip |
| Fire / Smoke | MIVIA Fire | two fire clips and two no-fire/fire-like clips |

## File conventions

```text
data/uploads/<descriptive_name>.mp4
data/ground_truth/<descriptive_name>_gt.json
data/ground_truth/benchmark_manifest.json
outputs/benchmark/<descriptive_name>_predictions.json
outputs/benchmark/benchmark_summary.json
```

Ground-truth format:

```json
{
  "video": "example.mp4",
  "segments": [
    { "start": 12.0, "end": 18.5, "label": "Running" }
  ]
}
```

## Work checklist

- [x] Remove generated clutter: duplicate UUID campus uploads, downloader
  sidecars, Python bytecode caches, and the corrupted pytest cache. Canonical
  MP4 clips, ground truth, source code, and benchmark outputs were retained.
- [x] Add per-video analyzer profiles to frontend and API.
- [x] Add benchmark manifest and batch benchmark command.
- [x] Run and record the initial local benchmark.
- [x] Add loitering, intrusion, and running as separate benchmark entries with proper zones.
- [x] Update README.md to reflect all 7 implemented analyzers.
- [x] Run tuned benchmark and inspect false positives/negatives.
- [x] Tune only measured thresholds/zones; rerun benchmark after every change (running: restricted zone, F1: 0.4 → 0.667).
- [x] Generate final timelines and report per-class metrics.
- [x] Add selected external clips to `data/uploads/`.
- [x] Convert image sequences to MP4 when required (URFD).
- [x] Trim each clip and create matching ground truth JSON.
- [x] Add each clip to `benchmark_manifest.json` with its activity profile and zone.
- [ ] Source Loitering and Intrusion benchmark videos (see Video Inventory section above).

## Commands

```powershell
# Start the app
python -m uvicorn app.main:app --reload --port 8000

# Run unit tests
python -m pytest tests -q

# Run the complete labelled benchmark
python scripts/run_benchmark.py

# Evaluate one saved prediction file
python scripts/run_eval.py --pred outputs/benchmark/<name>_predictions.json --gt data/ground_truth/<name>_gt.json --iou 0.3
```

## Guardrails

- Never use a full-frame zone as an intrusion zone unless every person truly
  counts as an intruder.
- Do not change a threshold without checking both positive and negative clips.
- Do not report a class as accurate until it has at least one positive and one
  negative evaluation clip.
- Keep Falling documented as a bbox-only heuristic until a pose/model-based
  confirmation stage is added.
