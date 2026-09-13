"""
Unit tests for activity analyzers and evaluation metrics.
"""

import pytest
from app.eval.temporal_iou import evaluate, evaluate_per_class, temporal_iou
from app.pipeline.aggregator import aggregate_segments, segments_to_dict
from app.pipeline.analyzers.base import Segment
from app.pipeline.analyzers.crowd import detect_crowd_formation
from app.pipeline.analyzers.intrusion import detect_intrusion
from app.pipeline.analyzers.loitering import detect_loitering
from app.pipeline.analyzers.running import detect_running
from app.pipeline.analyzer_runner import validate_activities
from app.pipeline.detect_track import TrackFrame


ZONE = [(100, 100), (500, 100), (500, 500), (100, 500)]


def test_activity_selection_validation():
    assert validate_activities(["running", "falling"]) == {"running", "falling"}
    with pytest.raises(ValueError):
        validate_activities(["fighting"])


def make_person_frame(frame_idx: int, timestamp: float, track_id: int, cx: float, cy: float, conf: float = 0.9) -> TrackFrame:
    """Helper to create a person TrackFrame with bbox centered at (cx, cy)."""
    return TrackFrame(
        frame_idx=frame_idx,
        timestamp=timestamp,
        track_id=track_id,
        cls="person",
        bbox=(cx - 20, cy - 40, cx + 20, cy + 40),
        conf=conf,
    )


def test_loitering_detected():
    # Person 1 stays inside ZONE from t=0.0 to t=15.0 (dwell = 15s >= 10s threshold)
    frames = [
        make_person_frame(idx, idx * 1.0, track_id=1, cx=200, cy=200)
        for idx in range(16)
    ]
    segments = detect_loitering(frames, ZONE, dwell_threshold_sec=10.0)
    assert len(segments) == 1
    assert segments[0].label == "Loitering"
    assert segments[0].start == 0.0
    assert segments[0].end == 15.0
    assert segments[0].track_ids == [1]
    assert segments[0].confidence > 0.5


def test_loitering_not_detected_short_dwell():
    # Person 1 inside ZONE only for 5s (< 10s threshold)
    frames = [
        make_person_frame(idx, idx * 1.0, track_id=1, cx=200, cy=200)
        for idx in range(6)
    ]
    segments = detect_loitering(frames, ZONE, dwell_threshold_sec=10.0)
    assert len(segments) == 0


def test_intrusion_detected():
    # Person 2 enters ZONE at t=5.0 and leaves at t=9.0
    frames = [
        make_person_frame(0, 0.0, track_id=2, cx=50, cy=50),     # outside
        make_person_frame(5, 5.0, track_id=2, cx=250, cy=250),   # inside
        make_person_frame(7, 7.0, track_id=2, cx=260, cy=260),   # inside
        make_person_frame(9, 9.0, track_id=2, cx=270, cy=270),   # inside
        make_person_frame(12, 12.0, track_id=2, cx=600, cy=600), # outside
    ]
    segments = detect_intrusion(frames, ZONE)
    assert len(segments) == 1
    assert segments[0].label == "Intrusion"
    assert segments[0].start == 5.0
    assert segments[0].end == 9.0
    assert segments[0].track_ids == [2]


def test_running_detected():
    # Person 3 moves rapidly: 200px in 0.5s -> 400 px/s (well above threshold of 120 px/s)
    frames = [
        make_person_frame(0, 0.0, track_id=3, cx=200, cy=200),
        make_person_frame(5, 0.5, track_id=3, cx=400, cy=200),
        make_person_frame(10, 1.0, track_id=3, cx=450, cy=200), # slows down
    ]
    segments = detect_running(frames, ZONE, velocity_threshold_px_sec=120.0, min_duration_sec=0.5)
    assert len(segments) == 1
    assert segments[0].label == "Running"
    assert segments[0].start == 0.0
    assert segments[0].track_ids == [3]


def test_crowd_formation_detected():
    # 4 distinct persons inside ZONE at t=10.0 and t=12.0
    frames = []
    for ts, idx in [(10.0, 100), (11.0, 110), (12.0, 120)]:
        for tid, (cx, cy) in enumerate([(200, 200), (250, 250), (300, 300), (350, 350)], start=1):
            frames.append(make_person_frame(idx, ts, track_id=tid, cx=cx, cy=cy))

    segments = detect_crowd_formation(frames, ZONE, count_threshold=3, min_duration_sec=1.0)
    assert len(segments) == 1
    assert segments[0].label == "Crowd Formation"
    assert segments[0].start == 10.0
    assert segments[0].end == 12.0
    assert set(segments[0].track_ids) == {1, 2, 3, 4}


def test_aggregator_and_eval():
    s1 = Segment(start=2.0, end=8.0, label="Intrusion", confidence=0.9, track_ids=[1])
    s2 = Segment(start=10.0, end=25.0, label="Loitering", confidence=0.85, track_ids=[2])
    merged = aggregate_segments([s2], [s1])
    assert [s.start for s in merged] == [2.0, 10.0]

    preds = segments_to_dict(merged)
    gt = [
        {"start": 2.5, "end": 8.0, "label": "Intrusion"},
        {"start": 12.0, "end": 24.0, "label": "Loitering"},
    ]
    results = evaluate_per_class(preds, gt, iou_threshold=0.3)
    assert "Intrusion" in results
    assert "Loitering" in results
    assert results["Intrusion"]["f1"] == 1.0
    assert results["Loitering"]["f1"] == 1.0


def make_object_frame(frame_idx: int, timestamp: float, track_id: int, cx: float, cy: float, cls_name: str = "backpack", conf: float = 0.85) -> TrackFrame:
    """Helper to create a luggage TrackFrame."""
    return TrackFrame(
        frame_idx=frame_idx,
        timestamp=timestamp,
        track_id=track_id,
        cls=cls_name,
        bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
        conf=conf,
    )


def test_abandoned_object_detected():
    from app.pipeline.analyzers.abandoned_object import detect_abandoned_object
    # Object 10 (backpack) stationary at (300, 300) from t=0 to t=8s
    # Person 1 walks away after t=0 (at t=1s, distance > 200px)
    frames = []
    for idx, ts in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
        frames.append(make_object_frame(idx, ts, track_id=10, cx=300, cy=300, cls_name="backpack"))
        # Person 1 moves far away
        frames.append(make_person_frame(idx, ts, track_id=1, cx=600 + idx * 20, cy=600))

    segments = detect_abandoned_object(frames, abandoned_threshold_sec=5.0)
    assert len(segments) == 1
    assert segments[0].label == "Abandoned Object"
    assert segments[0].track_ids == [10]
    assert segments[0].start == 0.0
    assert segments[0].end == 8.0


def test_abandoned_object_not_detected_owner_present():
    from app.pipeline.analyzers.abandoned_object import detect_abandoned_object
    # Person 1 stays right next to backpack (cx=310, cy=300, distance=10px < 120px)
    frames = []
    for idx, ts in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
        frames.append(make_object_frame(idx, ts, track_id=10, cx=300, cy=300))
        frames.append(make_person_frame(idx, ts, track_id=1, cx=310, cy=300))

    segments = detect_abandoned_object(frames, abandoned_threshold_sec=5.0)
    assert len(segments) == 0


def test_falling_detected():
    from app.pipeline.analyzers.falling import detect_falling
    # Person 5 starts upright: width=40, height=120 (AR=0.33, center_y=160)
    # Person 5 collapses: width=130, height=40 (AR=3.25, center_y=280) -> high vy and AR collapse
    frames = [
        TrackFrame(0, 0.0, 5, "person", (100, 100, 140, 220), 0.9),
        TrackFrame(5, 0.5, 5, "person", (80, 260, 210, 300), 0.85),
        TrackFrame(10, 1.0, 5, "person", (80, 260, 210, 300), 0.85),
        TrackFrame(13, 1.25, 5, "person", (80, 260, 210, 300), 0.85),
    ]
    segments = detect_falling(frames, settle_window_sec=0.75)
    assert len(segments) == 1
    assert segments[0].label == "Falling"
    assert segments[0].track_ids == [5]
    assert segments[0].start == 0.0
    assert segments[0].end == 1.25


def test_falling_not_detected_normal_walking():
    from app.pipeline.analyzers.falling import detect_falling
    # Person 6 walks with steady upright aspect ratio and constant y
    frames = [
        TrackFrame(0, 0.0, 6, "person", (100, 100, 140, 220), 0.9),
        TrackFrame(5, 0.5, 6, "person", (120, 102, 160, 222), 0.9),
        TrackFrame(10, 1.0, 6, "person", (140, 101, 180, 221), 0.9),
    ]
    segments = detect_falling(frames)
    assert len(segments) == 0


def test_falling_requires_both_signals_and_sustained_posture():
    from app.pipeline.analyzers.falling import detect_falling

    # Fast vertical movement with an upright bbox is not a fall.
    fast_upright = [
        TrackFrame(0, 0.0, 7, "person", (100, 100, 140, 220), 0.9),
        TrackFrame(5, 0.5, 7, "person", (100, 240, 140, 360), 0.9),
        TrackFrame(13, 1.25, 7, "person", (100, 240, 140, 360), 0.9),
    ]
    # A sharp flip that returns upright during settling is a bend/stumble.
    recovers = [
        TrackFrame(0, 0.0, 8, "person", (100, 100, 140, 220), 0.9),
        TrackFrame(5, 0.5, 8, "person", (80, 260, 210, 300), 0.9),
        TrackFrame(10, 1.0, 8, "person", (100, 280, 140, 400), 0.9),
        TrackFrame(13, 1.25, 8, "person", (100, 280, 140, 400), 0.9),
    ]
    assert detect_falling(fast_upright) == []
    assert detect_falling(recovers) == []


def test_fire_smoke_chroma_detection():
    import numpy as np
    from app.pipeline.analyzers.fire_smoke import detect_fire_smoke_chroma

    # Normal green/blue scene -> no fire
    normal_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    normal_frame[:, :] = (100, 150, 50)
    is_fire, _ = detect_fire_smoke_chroma(normal_frame)
    assert not is_fire

    # Frame with bright red/orange fire patch
    fire_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    fire_frame[50:100, 50:100] = (20, 140, 255) # BGR orange-red fire color
    is_fire, conf = detect_fire_smoke_chroma(fire_frame)
    assert is_fire
    assert conf > 0.5
