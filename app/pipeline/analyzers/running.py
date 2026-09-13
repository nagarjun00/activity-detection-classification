"""
Running in restricted zone analyzer.

Rule: compute centroid velocity between consecutive frames for person tracks
inside a zone polygon. If the velocity exceeds a threshold (in pixels/second)
for at least a minimum duration, flag as "Running".
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.pipeline.analyzers.base import Segment, bbox_center, point_in_zone
from app.pipeline.detect_track import TrackFrame

DEFAULT_VELOCITY_THRESHOLD_PX_SEC = 120.0  # pixels per second
DEFAULT_MIN_DURATION_SEC = 0.5            # minimum duration to filter momentary spikes
DEFAULT_MAX_GAP_SEC = 1.5                 # gap tolerance in seconds


def detect_running(
    frames: List[TrackFrame],
    zone_polygon: Optional[List[Tuple[float, float]]] = None,
    velocity_threshold_px_sec: float = DEFAULT_VELOCITY_THRESHOLD_PX_SEC,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> List[Segment]:
    """
    Detects running segments for person tracks whose velocity exceeds threshold.

    frames: list of TrackFrame instances from the tracker
    zone_polygon: optional polygon to restrict detection zone. If None, checks full frame.
    velocity_threshold_px_sec: velocity threshold in pixels per second.
    min_duration_sec: minimum duration to qualify as a sustained running event.
    max_gap_sec: maximum gap allowed between running detections before splitting.
    """
    track_history: Dict[int, List[TrackFrame]] = defaultdict(list)

    for f in frames:
        if f.cls != "person":
            continue
        cx, cy = bbox_center(f.bbox)
        if zone_polygon is None or point_in_zone(cx, cy, zone_polygon):
            track_history[f.track_id].append(f)

    segments: List[Segment] = []

    for track_id, hist in track_history.items():
        hist.sort(key=lambda x: x.timestamp)
        if len(hist) < 2:
            continue

        # Find frames with high instantaneous velocity
        running_frames: List[Tuple[TrackFrame, float]] = []  # (frame, velocity)
        for i in range(1, len(hist)):
            prev_f = hist[i - 1]
            curr_f = hist[i]
            dt = curr_f.timestamp - prev_f.timestamp
            if dt <= 0:
                continue

            cx1, cy1 = bbox_center(prev_f.bbox)
            cx2, cy2 = bbox_center(curr_f.bbox)
            dist = math.hypot(cx2 - cx1, cy2 - cy1)
            vel = dist / dt

            if vel >= velocity_threshold_px_sec:
                if not running_frames or running_frames[-1][0].frame_idx != prev_f.frame_idx:
                    running_frames.append((prev_f, vel))
                running_frames.append((curr_f, vel))

        if not running_frames:
            continue

        # Merge consecutive running frames into segments
        run_group: List[Tuple[TrackFrame, float]] = [running_frames[0]]
        prev_time = running_frames[0][0].timestamp

        def close_run(group: List[Tuple[TrackFrame, float]]):
            if not group:
                return
            start = group[0][0].timestamp
            end = group[-1][0].timestamp
            duration = end - start
            if duration >= min_duration_sec:
                # Average detection confidence and velocity boost
                avg_conf = sum(f[0].conf for f in group) / len(group)
                avg_vel = sum(f[1] for f in group) / len(group)
                vel_factor = min(1.0, avg_vel / (velocity_threshold_px_sec * 1.5))
                confidence = min(1.0, (avg_conf * 0.7) + (vel_factor * 0.3))

                segments.append(
                    Segment(
                        start=round(start, 2),
                        end=round(end, 2),
                        label="Running",
                        confidence=round(confidence, 2),
                        track_ids=[track_id],
                    )
                )

        for item in running_frames[1:]:
            tf, vel = item
            if tf.timestamp - prev_time > max_gap_sec:
                close_run(run_group)
                run_group = [item]
            else:
                run_group.append(item)
            prev_time = tf.timestamp

        close_run(run_group)

    segments.sort(key=lambda s: s.start)
    return segments
