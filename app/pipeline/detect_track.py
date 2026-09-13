"""
Detection & tracking module.

Runs a YOLO model over a video with built-in ByteTrack tracking (via
ultralytics' `model.track()`), and returns a flat list of per-frame,
per-track detections that every downstream analyzer consumes.

This is the reused core from the ANPR/traffic project: YOLO detection +
object tracking. Here it defaults to tracking people (COCO class 0),
since most of the target activity classes (loitering, intrusion, running,
crowd formation, abandoned object) hinge on person and object tracks over
time rather than single-frame classification.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO

# Pretrained COCO weights. Swap for a fine-tuned checkpoint later if needed.
MODEL_PATH = "yolov8n.pt"

# COCO class indices relevant to this assignment.
# 0 = person (loitering, intrusion, running, crowd, falling, fighting)
# Objects for "abandoned object" (bag/backpack/suitcase) are added on top
# separately since they need a different tracking rule (stationary + no
# owner nearby), see app/pipeline/analyzers/abandoned_object.py
COCO_PERSON = 0
COCO_OBJECTS_OF_INTEREST = [24, 26, 28]  # backpack, handbag, suitcase


@dataclass
class TrackFrame:
    frame_idx: int
    timestamp: float  # seconds, computed from frame_idx / fps
    track_id: int
    cls: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    conf: float


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 25.0


def run_detection_tracking(
    video_path: str,
    classes: Optional[List[int]] = None,
    model_path: str = MODEL_PATH,
) -> List[TrackFrame]:
    """
    Runs YOLO + ByteTrack over the full video and returns a flat list of
    TrackFrame records, one per detected+tracked object per frame.

    classes: COCO class indices to keep. Defaults to person only.
    """
    if classes is None:
        classes = [COCO_PERSON] + COCO_OBJECTS_OF_INTEREST

    fps = get_video_fps(video_path)
    model = YOLO(model_path)

    # stream=True keeps this memory-safe for long videos: frames are
    # yielded one at a time instead of held in memory all at once.
    results = model.track(
        source=video_path,
        classes=classes,
        persist=True,
        stream=True,
        verbose=False,
    )

    frames: List[TrackFrame] = []
    for frame_idx, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            # No tracked detections in this frame — skip.
            continue

        boxes = result.boxes
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            cls_name = model.names[cls_id]
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            timestamp = frame_idx / fps

            frames.append(
                TrackFrame(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    track_id=track_id,
                    cls=cls_name,
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                )
            )

    return frames
