"""
Abandoned Object analyzer.

Rule: an object track (backpack, handbag, suitcase) whose bounding box centroid
remains stationary (within a small pixel radius) for >= abandoned_threshold_sec,
with NO person track centroid within a distance threshold (owner proximity)
for that same duration.

Confidence scales with duration beyond the threshold, capped at 1.0.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from app.pipeline.analyzers.base import Segment, bbox_center
from app.pipeline.detect_track import TrackFrame

LUGGAGE_CLASSES = {"backpack", "handbag", "suitcase"}
DEFAULT_ABANDONED_THRESHOLD_SEC = 5.0  # seconds unattended & stationary
DEFAULT_MAX_STATIONARY_DRIFT_PX = 100.0  # max centroid drift to be considered stationary
DEFAULT_OWNER_PROXIMITY_PX = 120.0     # radius within which a person is considered the owner
DEFAULT_MAX_GAP_SEC = 2.0              # gap tolerance between detections


def detect_abandoned_object(
    frames: List[TrackFrame],
    abandoned_threshold_sec: float = DEFAULT_ABANDONED_THRESHOLD_SEC,
    stationary_drift_px: float = DEFAULT_MAX_STATIONARY_DRIFT_PX,
    owner_proximity_px: float = DEFAULT_OWNER_PROXIMITY_PX,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> List[Segment]:
    """
    Detects abandoned luggage objects separated from any owner.

    frames: list of TrackFrame instances containing both person and object detections.
    abandoned_threshold_sec: minimum seconds unattended and stationary.
    stationary_drift_px: maximum drift radius in pixels for the object to count as stationary.
    owner_proximity_px: minimum distance to the nearest person to be considered 'unattended'.
    max_gap_sec: tolerance for missed detections.
    """
    # Separate person frames and object frames by timestamp
    object_history: Dict[int, List[TrackFrame]] = defaultdict(list)
    person_by_timestamp: Dict[float, List[Tuple[float, float]]] = defaultdict(list)

    for f in frames:
        cx, cy = bbox_center(f.bbox)
        if f.cls == "person":
            person_by_timestamp[f.timestamp].append((cx, cy))
        elif f.cls in LUGGAGE_CLASSES:
            object_history[f.track_id].append(f)

    segments: List[Segment] = []

    for track_id, hist in object_history.items():
        hist.sort(key=lambda x: x.timestamp)
        if not hist:
            continue

        # Evaluate candidate unattended runs
        anchor_cx, anchor_cy = bbox_center(hist[0].bbox)
        run_frames: List[TrackFrame] = []
        prev_time = hist[0].timestamp

        def close_run(run: List[TrackFrame]):
            if not run:
                return
            start = run[0].timestamp
            end = run[-1].timestamp
            duration = end - start
            if duration >= abandoned_threshold_sec:
                # Confidence scales with how long the object has been left unattended
                confidence = min(1.0, duration / (abandoned_threshold_sec * 2))
                avg_det_conf = sum(f.conf for f in run) / len(run)
                final_conf = min(1.0, (avg_det_conf * 0.5) + (confidence * 0.5))

                segments.append(
                    Segment(
                        start=round(start, 2),
                        end=round(end, 2),
                        label="Abandoned Object",
                        confidence=round(final_conf, 2),
                        track_ids=[track_id],
                    )
                )

        for tf in hist:
            cx, cy = bbox_center(tf.bbox)
            drift = math.hypot(cx - anchor_cx, cy - anchor_cy)

            # Check if stationary
            if drift > stationary_drift_px:
                # Object moved significantly — reset anchor
                close_run(run_frames)
                run_frames = []
                anchor_cx, anchor_cy = cx, cy

            # Check if any person is within owner proximity at this timestamp
            nearby_persons = person_by_timestamp.get(tf.timestamp, [])
            owner_present = any(
                math.hypot(cx - px, cy - py) <= owner_proximity_px
                for px, py in nearby_persons
            )

            if owner_present:
                close_run(run_frames)
                run_frames = []
            else:
                if run_frames and (tf.timestamp - prev_time > max_gap_sec):
                    close_run(run_frames)
                    run_frames = [tf]
                else:
                    run_frames.append(tf)
                prev_time = tf.timestamp

        close_run(run_frames)

    segments.sort(key=lambda s: s.start)
    return segments
