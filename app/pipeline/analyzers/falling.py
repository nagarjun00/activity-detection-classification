"""Bounding-box-only falling analyzer.

A fall candidate requires both a rapid increase in bbox width/height and a
downward centroid movement in the same look-back window. It is emitted only
when the widened posture remains present for a short settle window.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from app.pipeline.analyzers.base import Segment, bbox_center
from app.pipeline.detect_track import TrackFrame


# Heuristic starting points for 1080p-like footage; tune per target camera.
DEFAULT_ASPECT_RATIO_DELTA_THRESHOLD = 0.6
DEFAULT_VERTICAL_VELOCITY_THRESHOLD_PX_SEC = 150.0
DEFAULT_SETTLE_WINDOW_SEC = 0.75
DEFAULT_WINDOW_SEC = 1.0


def get_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """Return bbox width divided by height, guarding against zero height."""
    x1, y1, x2, y2 = bbox
    return max(1.0, x2 - x1) / max(1.0, y2 - y1)


def detect_falling(
    frames: List[TrackFrame],
    aspect_ratio_delta_threshold: float = DEFAULT_ASPECT_RATIO_DELTA_THRESHOLD,
    vertical_velocity_threshold: float = DEFAULT_VERTICAL_VELOCITY_THRESHOLD_PX_SEC,
    settle_window_sec: float = DEFAULT_SETTLE_WINDOW_SEC,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> List[Segment]:
    """Detect sustained falls from person bbox history.

    A candidate must exceed *both* thresholds in one look-back window. It must
    then retain at least half of the aspect-ratio flip through the settle
    window. Confidence is a capped signal-strength heuristic, not a calibrated
    probability.
    """
    if aspect_ratio_delta_threshold <= 0 or vertical_velocity_threshold <= 0:
        raise ValueError("fall thresholds must be positive")
    if settle_window_sec <= 0 or window_sec <= 0:
        raise ValueError("fall windows must be positive")

    track_history: Dict[int, List[TrackFrame]] = defaultdict(list)
    for frame in frames:
        if frame.cls == "person":
            track_history[frame.track_id].append(frame)

    segments: List[Segment] = []
    for track_id, history in track_history.items():
        history.sort(key=lambda frame: frame.timestamp)
        if len(history) < 3:
            continue

        last_emitted_end = float("-inf")
        for end_index, end_frame in enumerate(history):
            # Oldest observation in the time window keeps this stable when
            # tracker sampling is irregular.
            start_index = end_index
            while (
                start_index > 0
                and end_frame.timestamp - history[start_index - 1].timestamp <= window_sec
            ):
                start_index -= 1

            start_frame = history[start_index]
            dt = end_frame.timestamp - start_frame.timestamp
            if dt <= 0 or start_frame.timestamp < last_emitted_end:
                continue

            start_ar = get_aspect_ratio(start_frame.bbox)
            end_ar = get_aspect_ratio(end_frame.bbox)
            aspect_delta = end_ar - start_ar
            _, start_y = bbox_center(start_frame.bbox)
            _, end_y = bbox_center(end_frame.bbox)
            vertical_velocity = (end_y - start_y) / dt

            if (
                aspect_delta < aspect_ratio_delta_threshold
                or vertical_velocity < vertical_velocity_threshold
            ):
                continue

            settle_end = end_frame.timestamp + settle_window_sec
            settle_frames = []
            observed_through_settle_end = False
            for frame in history[end_index:]:
                if frame.timestamp > settle_end:
                    observed_through_settle_end = True
                    break
                settle_frames.append(frame)
                if frame.timestamp == settle_end:
                    observed_through_settle_end = True
                    break

            # Video sampling rarely lands exactly on the requested boundary.
            # Seeing the next frame after it proves the final in-window frame
            # covers the complete settle period.
            if not settle_frames or not observed_through_settle_end:
                continue

            elevated_aspect_ratio = start_ar + (aspect_delta * 0.5)
            if any(
                get_aspect_ratio(frame.bbox) < elevated_aspect_ratio
                for frame in settle_frames
            ):
                continue

            signal_strength = (
                aspect_delta / aspect_ratio_delta_threshold
                + vertical_velocity / vertical_velocity_threshold
            ) / 3.0
            segments.append(
                Segment(
                    start=round(start_frame.timestamp, 2),
                    end=round(settle_end, 2),
                    label="Falling",
                    confidence=round(min(1.0, signal_strength), 2),
                    track_ids=[track_id],
                )
            )
            last_emitted_end = settle_end

    segments.sort(key=lambda segment: segment.start)
    return segments
