"""
Loitering analyzer.

Rule: a tracked person is "loitering" if they remain inside a defined zone
for longer than a dwell-time threshold, allowing brief gaps (missed
detections) without breaking the run.

This is the simplest analyzer and the template for the zone-based ones
(intrusion, running-in-restricted-zone, crowd formation) — see the TODO
list in README.md for the remaining analyzers to write following this
same pattern.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from app.pipeline.analyzers.base import Segment, bbox_center, point_in_zone
from app.pipeline.detect_track import TrackFrame

DEFAULT_DWELL_THRESHOLD_SEC = 10.0
DEFAULT_MAX_GAP_SEC = 1.5  # tolerance for missed frames within one dwell run


def detect_loitering(
    frames: List[TrackFrame],
    zone_polygon: List[Tuple[float, float]],
    dwell_threshold_sec: float = DEFAULT_DWELL_THRESHOLD_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> List[Segment]:
    track_history: Dict[int, List[TrackFrame]] = defaultdict(list)

    for f in frames:
        if f.cls != "person":
            continue
        cx, cy = bbox_center(f.bbox)
        if point_in_zone(cx, cy, zone_polygon):
            track_history[f.track_id].append(f)

    segments: List[Segment] = []

    for track_id, hist in track_history.items():
        hist.sort(key=lambda x: x.timestamp)
        if not hist:
            continue

        run_start = hist[0].timestamp
        prev_time = hist[0].timestamp

        def close_run(start: float, end: float):
            duration = end - start
            if duration >= dwell_threshold_sec:
                # Confidence scales with how far past the threshold the
                # dwell went, capped at 1.0. This is a simple heuristic —
                # documented as a design choice in the report, not a
                # calibrated probability.
                confidence = min(1.0, duration / (dwell_threshold_sec * 2))
                segments.append(
                    Segment(
                        start=start,
                        end=end,
                        label="Loitering",
                        confidence=round(confidence, 2),
                        track_ids=[track_id],
                    )
                )

        for tf in hist[1:]:
            if tf.timestamp - prev_time > max_gap_sec:
                close_run(run_start, prev_time)
                run_start = tf.timestamp
            prev_time = tf.timestamp

        close_run(run_start, prev_time)

    return segments
