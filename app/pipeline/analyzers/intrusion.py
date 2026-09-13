"""
Intrusion / Trespassing analyzer.

Rule: a tracked person is flagged for "Intrusion" whenever their centroid enters
a restricted zone polygon. Unlike loitering, intrusion does not require a minimum
dwell time — any entry into the restricted zone triggers a detection segment.
Consecutive detections within the zone (tolerating brief gaps) are merged into
a continuous intrusion interval.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from app.pipeline.analyzers.base import Segment, bbox_center, point_in_zone
from app.pipeline.detect_track import TrackFrame

DEFAULT_MAX_GAP_SEC = 2.5  # tolerance for missed frames within one intrusion run


def detect_intrusion(
    frames: List[TrackFrame],
    zone_polygon: List[Tuple[float, float]],
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> List[Segment]:
    """
    Detects intrusion segments for person tracks inside zone_polygon.

    frames: list of TrackFrame instances from the tracker
    zone_polygon: list of (x, y) coordinates defining the restricted zone
    max_gap_sec: maximum gap in seconds before breaking an ongoing intrusion run
    """
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

        run_frames: List[TrackFrame] = [hist[0]]
        prev_time = hist[0].timestamp

        def close_run(run: List[TrackFrame]):
            if not run:
                return
            start = run[0].timestamp
            end = run[-1].timestamp
            # Average detection confidence across the intrusion run
            avg_conf = sum(f.conf for f in run) / len(run) if run else 0.8
            segments.append(
                Segment(
                    start=round(start, 2),
                    end=round(end, 2),
                    label="Intrusion",
                    confidence=round(avg_conf, 2),
                    track_ids=[track_id],
                )
            )

        for tf in hist[1:]:
            if tf.timestamp - prev_time > max_gap_sec:
                close_run(run_frames)
                run_frames = [tf]
            else:
                run_frames.append(tf)
            prev_time = tf.timestamp

        close_run(run_frames)

    segments.sort(key=lambda s: s.start)
    return segments
