"""
Video utility helpers for CCTV clip downloading, trimming, and frame extraction.
"""

import os
import subprocess
import cv2
from typing import Dict, Optional


def get_video_info(video_path: str) -> Dict:
    """Returns video duration, fps, resolution, and total frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "video_path": video_path,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "resolution": (width, height),
        "duration_sec": round(duration, 2),
    }


def trim_video_cv2(input_path: str, output_path: str, start_sec: float, end_sec: float) -> str:
    """Trims a video using OpenCV VideoCapture and VideoWriter (no external ffmpeg required)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    return output_path


def extract_key_frames(video_path: str, output_dir: str, interval_sec: float = 5.0) -> int:
    """Extracts snapshot frames at regular intervals to inspect activity and note ground truth."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(interval_sec * fps))

    saved_count = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = round(frame_idx / fps, 2)
            out_name = os.path.join(output_dir, f"frame_{timestamp:06.2f}s.jpg")
            cv2.imwrite(out_name, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    return saved_count


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        info = get_video_info(sys.argv[1])
        print(info)
