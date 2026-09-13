"""
Crowd Formation analyzer.

Rule: count the number of distinct active person tracks located inside a zone
over time. When the simultaneous count of people inside the zone meets or exceeds
a threshold (e.g. >= 3 or 4 persons) for at least a minimum duration, flag as
"Crowd Formation".
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from app.pipeline.analyzers.base import Segment, bbox_center, point_in_zone
from app.pipeline.detect_track import TrackFrame

DEFAULT_CROWD_COUNT_THRESHOLD = 3   # minimum people to form a crowd
DEFAULT_MIN_DURATION_SEC = 1.0      # minimum duration to qualify
DEFAULT_MAX_GAP_SEC = 1.5           # maximum gap between consecutive crowd timestamps


def detect_crowd_formation(
    frames: List[TrackFrame],
    zone_polygon: Optional[List[Tuple[float, float]]] = None,
    count_threshold: int = DEFAULT_CROWD_COUNT_THRESHOLD,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> List[Segment]:
    """
    Detects crowd formation segments where person density in a zone exceeds count_threshold.

    frames: list of TrackFrame instances from the tracker
    zone_polygon: optional polygon to restrict detection zone. If None, checks full frame.
    count_threshold: minimum number of simultaneous persons inside the zone.
    min_duration_sec: minimum sustained duration in seconds.
    max_gap_sec: maximum gap allowed between crowd time points before splitting segments.
    """
    # Group detections by timestamp
    timestamp_tracks: Dict[float, Set[int]] = defaultdict(set)
    timestamp_frames: Dict[float, List[TrackFrame]] = defaultdict(list)

    for f in frames:
        if f.cls != "person":
            continue
        cx, cy = bbox_center(f.bbox)
        if zone_polygon is None or point_in_zone(cx, cy, zone_polygon):
            timestamp_tracks[f.timestamp].add(f.track_id)
            timestamp_frames[f.timestamp].append(f)

    # Identify timestamps meeting crowd threshold
    crowd_timestamps = sorted(
        ts for ts, tracks in timestamp_tracks.items() if len(tracks) >= count_threshold
    )

    if not crowd_timestamps:
        return []

    segments: List[Segment] = []
    run_timestamps = [crowd_timestamps[0]]
    prev_time = crowd_timestamps[0]

    def close_run(ts_list: List[float]):
        if not ts_list:
            return
        start = ts_list[0]
        end = ts_list[-1]
        duration = end - start
        if duration >= min_duration_sec or len(ts_list) == 1:
            all_tracks: Set[int] = set()
            all_confs: List[float] = []
            max_count = 0

            for ts in ts_list:
                tracks = timestamp_tracks[ts]
                all_tracks.update(tracks)
                max_count = max(max_count, len(tracks))
                for f in timestamp_frames[ts]:
                    all_confs.append(f.conf)

            avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.8
            # Confidence scales with crowd size above threshold
            density_factor = min(1.0, max_count / (count_threshold * 1.5))
            confidence = min(1.0, (avg_conf * 0.6) + (density_factor * 0.4))

            segments.append(
                Segment(
                    start=round(start, 2),
                    end=round(end, 2),
                    label="Crowd Formation",
                    confidence=round(confidence, 2),
                    track_ids=sorted(list(all_tracks)),
                )
            )

    for ts in crowd_timestamps[1:]:
        if ts - prev_time > max_gap_sec:
            close_run(run_timestamps)
            run_timestamps = [ts]
        else:
            run_timestamps.append(ts)
        prev_time = ts

    close_run(run_timestamps)

    segments.sort(key=lambda s: s.start)
    return segments
