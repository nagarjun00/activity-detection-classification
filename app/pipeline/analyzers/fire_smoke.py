"""
Fire / Smoke analyzer.

Appearance-based detector that analyzes video frames directly (not tracking-based).
Samples frames from the video, applies fire/smoke detection, and converts
runs of consecutive positive frames into continuous activity segments.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from app.pipeline.analyzers.base import Segment

DEFAULT_SAMPLE_RATE_FPS = 4.0      # Sample 4 frames per second for speed
DEFAULT_MIN_FIRE_CONSECUTIVE = 2   # Minimum consecutive positive detections
DEFAULT_MIN_DURATION_SEC = 0.5     # Minimum duration of fire/smoke event


def detect_fire_smoke_chroma(frame: np.ndarray) -> Tuple[bool, float]:
    """
    Chromaticity and luminance analysis for fire and smoke.
    Fire exhibits characteristic high red-yellow intensity:
    In RGB: R > G > B, and in YCbCr: Y >= Cr >= Cb with high chrominance difference.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Fire color range in HSV: Hue [0-25] and [165-180], high Saturation & Value
    lower_fire1 = np.array([0, 120, 160], dtype=np.uint8)
    upper_fire1 = np.array([28, 255, 255], dtype=np.uint8)
    lower_fire2 = np.array([170, 120, 160], dtype=np.uint8)
    upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    fire_mask = cv2.bitwise_or(mask1, mask2)

    # Smoke: grayish low-saturation moderate-value diffuse regions
    lower_smoke = np.array([0, 0, 100], dtype=np.uint8)
    upper_smoke = np.array([180, 50, 200], dtype=np.uint8)
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

    total_pixels = frame.shape[0] * frame.shape[1]
    fire_pixels = cv2.countNonZero(fire_mask)
    smoke_pixels = cv2.countNonZero(smoke_mask)

    fire_ratio = fire_pixels / total_pixels
    smoke_ratio = smoke_pixels / total_pixels

    # Detection condition: significant cluster of fire pixels (>= 0.05% of frame)
    # or combined fire + smoke plume
    if fire_ratio >= 0.0008:  # 0.08% of frame
        conf = min(1.0, 0.65 + (fire_ratio * 50.0))
        return True, round(conf, 2)
    elif fire_ratio >= 0.0003 and smoke_ratio >= 0.02:
        conf = min(1.0, 0.55 + (fire_ratio * 40.0))
        return True, round(conf, 2)

    return False, 0.0


def detect_fire_smoke(
    video_path: str,
    sample_rate_fps: float = DEFAULT_SAMPLE_RATE_FPS,
    min_consecutive: int = DEFAULT_MIN_FIRE_CONSECUTIVE,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> List[Segment]:
    """
    Runs appearance-based fire/smoke detection over sampled frames in video_path.
    Returns List[Segment] with track_ids=[].
    """
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(native_fps / sample_rate_fps)))

    positive_timestamps: List[Tuple[float, float]] = []  # (timestamp, confidence)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = frame_idx / native_fps
            is_detected, conf = detect_fire_smoke_chroma(frame)
            if is_detected:
                positive_timestamps.append((timestamp, conf))

        frame_idx += 1

    cap.release()

    if not positive_timestamps:
        return []

    # Cluster consecutive positive timestamps into continuous segments
    segments: List[Segment] = []
    max_gap_sec = (frame_step / native_fps) * 2.5  # gap tolerance

    run_start = positive_timestamps[0][0]
    run_end = positive_timestamps[0][0]
    run_confs = [positive_timestamps[0][1]]
    prev_time = positive_timestamps[0][0]

    def close_run(start: float, end: float, confs: List[float]):
        duration = end - start
        if len(confs) >= min_consecutive or duration >= min_duration_sec:
            avg_conf = sum(confs) / len(confs) if confs else 0.75
            segments.append(
                Segment(
                    start=round(start, 2),
                    end=round(end, 2),
                    label="Fire / Smoke",
                    confidence=round(avg_conf, 2),
                    track_ids=[],  # Appearance-based, no track IDs
                )
            )

    for ts, conf in positive_timestamps[1:]:
        if ts - prev_time > max_gap_sec:
            close_run(run_start, run_end, run_confs)
            run_start = ts
            run_end = ts
            run_confs = [conf]
        else:
            run_end = ts
            run_confs.append(conf)
        prev_time = ts

    close_run(run_start, run_end, run_confs)
    segments.sort(key=lambda s: s.start)
    return segments
