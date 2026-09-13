"""
Shared types for all per-class analyzers.

Every analyzer takes the flat List[TrackFrame] from detect_track.py and
returns List[Segment] — this is the common contract the aggregator relies
on, so adding a new activity class means writing one function with this
signature and dropping it into main.py's pipeline call.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Segment:
    start: float          # seconds
    end: float             # seconds
    label: str              # activity class name
    confidence: float       # 0.0-1.0
    track_ids: List[int] = field(default_factory=list)  # contributing track IDs


def point_in_zone(cx: float, cy: float, zone_polygon: List[tuple]) -> bool:
    """
    Ray-casting point-in-polygon test. Used by every zone-based analyzer
    (loitering, intrusion, running-in-restricted-zone, crowd formation).

    zone_polygon: list of (x, y) pixel coordinates defining the region.
    """
    n = len(zone_polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = zone_polygon[i]
        xj, yj = zone_polygon[j]
        if ((yi > cy) != (yj > cy)) and (
            cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def bbox_center(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2
