"""
Segment aggregator.

Takes the List[Segment] output from every analyzer that ran, merges them
into one sorted list, and serializes to the plain-dict shape the API
returns and the eval module consumes.
"""

from typing import Dict, List

from app.pipeline.analyzers.base import Segment


def aggregate_segments(*segment_lists: List[Segment]) -> List[Segment]:
    all_segments: List[Segment] = [s for lst in segment_lists for s in lst]
    all_segments.sort(key=lambda s: s.start)
    return all_segments


def segments_to_dict(segments: List[Segment]) -> List[Dict]:
    return [
        {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "label": s.label,
            "confidence": s.confidence,
            "track_ids": s.track_ids,
        }
        for s in segments
    ]
