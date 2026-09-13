"""
Temporal evaluation for activity segments.

Standard metric for temporal action detection: match predicted segments
to ground-truth segments of the same class using temporal Intersection
over Union (IoU), then report precision/recall/F1 at a given IoU
threshold. This mirrors the same IoU concept used for bounding-box
detection (mAP/IoU) applied to time intervals instead of spatial boxes —
worth stating explicitly in the report since it's the metric-choice
justification the brief asks for.
"""

from typing import Dict, List


def temporal_iou(pred: Dict, gt: Dict) -> float:
    start = max(pred["start"], gt["start"])
    end = min(pred["end"], gt["end"])
    intersection = max(0.0, end - start)
    union = max(pred["end"], gt["end"]) - min(pred["start"], gt["start"])
    return intersection / union if union > 0 else 0.0


def evaluate(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.3,
) -> Dict:
    """
    Greedy matching: for each prediction, find the best unmatched
    ground-truth segment of the same class. A match counts as a true
    positive if its IoU clears iou_threshold.

    iou_threshold=0.3 (rather than the 0.5 common in spatial detection)
    is a deliberate, documented choice: activity boundaries are inherently
    fuzzier to annotate by hand than object bounding boxes, so a stricter
    0.5 threshold would penalize reasonable predictions for boundary
    disagreement rather than genuine misses. State this reasoning in the
    report.
    """
    matched_gt = set()
    tp = 0

    for p in predictions:
        best_iou = 0.0
        best_gt_idx = None
        for idx, g in enumerate(ground_truth):
            if idx in matched_gt or g["label"] != p["label"]:
                continue
            iou = temporal_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou_threshold": iou_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def evaluate_per_class(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.3,
) -> Dict[str, Dict]:
    """Runs `evaluate` separately per activity class, for the report's
    per-class breakdown table."""
    classes = sorted({s["label"] for s in ground_truth} | {s["label"] for s in predictions})
    results = {}
    for cls in classes:
        cls_preds = [p for p in predictions if p["label"] == cls]
        cls_gt = [g for g in ground_truth if g["label"] == cls]
        results[cls] = evaluate(cls_preds, cls_gt, iou_threshold)
    return results
