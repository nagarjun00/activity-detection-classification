"""Run a selected, camera-appropriate subset of activity analyzers.

Not every detector is meaningful for every video.  For example, a walkway
clip should normally run running/crowd/intrusion rules, while a fire clip
should run only fire/smoke.  Selecting activities prevents unrelated weak
heuristics from polluting a video's result set.
"""

from typing import Iterable, List, Optional, Sequence, Set, Tuple

from app.pipeline.analyzers.abandoned_object import detect_abandoned_object
from app.pipeline.analyzers.base import Segment
from app.pipeline.analyzers.crowd import detect_crowd_formation
from app.pipeline.analyzers.falling import detect_falling
from app.pipeline.analyzers.fire_smoke import detect_fire_smoke
from app.pipeline.analyzers.intrusion import detect_intrusion
from app.pipeline.analyzers.loitering import detect_loitering
from app.pipeline.analyzers.running import detect_running
from app.pipeline.detect_track import TrackFrame


ACTIVITIES: Set[str] = {
    "loitering",
    "intrusion",
    "running",
    "crowd",
    "abandoned_object",
    "falling",
    "fire_smoke",
}
TRACK_BASED_ACTIVITIES: Set[str] = ACTIVITIES - {"fire_smoke"}


def validate_activities(activities: Optional[Iterable[str]]) -> Set[str]:
    """Return a validated activity selection; ``None`` means all analyzers."""
    if activities is None:
        return set(ACTIVITIES)
    selected = {activity.strip().lower() for activity in activities}
    unknown = selected - ACTIVITIES
    if unknown:
        raise ValueError(f"Unknown activities: {', '.join(sorted(unknown))}")
    return selected


def run_selected_analyzers(
    frames: Sequence[TrackFrame],
    video_path: str,
    zone_polygon: List[Tuple[float, float]],
    activities: Optional[Iterable[str]] = None,
    loitering_dwell_threshold_sec: float = 10.0,
) -> List[Segment]:
    """Run only the requested analyzer classes and return sorted segments."""
    selected = validate_activities(activities)
    segments: List[Segment] = []

    if "loitering" in selected:
        segments.extend(
            detect_loitering(
                list(frames), zone_polygon, dwell_threshold_sec=loitering_dwell_threshold_sec
            )
        )
    if "intrusion" in selected:
        segments.extend(detect_intrusion(list(frames), zone_polygon))
    if "running" in selected:
        segments.extend(detect_running(list(frames), zone_polygon))
    if "crowd" in selected:
        segments.extend(detect_crowd_formation(list(frames), zone_polygon))
    if "abandoned_object" in selected:
        segments.extend(detect_abandoned_object(list(frames)))
    if "falling" in selected:
        segments.extend(detect_falling(list(frames)))
    if "fire_smoke" in selected:
        segments.extend(detect_fire_smoke(video_path))

    segments.sort(key=lambda segment: segment.start)
    return segments
