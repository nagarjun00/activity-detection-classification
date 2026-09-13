"""
Command-line evaluation runner for temporal activity detection.

Usage:
  python scripts/run_eval.py --pred outputs/campus_walk_30s_predictions.json --gt data/ground_truth/campus_walk_30s_gt.json --iou 0.3
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.eval.temporal_iou import evaluate_per_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted activity segments against ground truth.")
    parser.add_argument("--pred", required=True, help="Path to predicted segments JSON file")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--iou", type=float, default=0.3, help="Temporal IoU threshold (default: 0.3)")
    args = parser.parse_args()

    if not os.path.exists(args.pred):
        print(f"Error: Prediction file not found: {args.pred}")
        return
    if not os.path.exists(args.gt):
        print(f"Error: Ground truth file not found: {args.gt}")
        return

    with open(args.pred, "r") as f:
        pred_data = json.load(f)
        preds = pred_data.get("segments", pred_data)

    with open(args.gt, "r") as f:
        gt_data = json.load(f)
        gt = gt_data.get("segments", gt_data)

    results = evaluate_per_class(preds, gt, iou_threshold=args.iou)

    print("\n" + "=" * 65)
    print(f"TEMPORAL EVALUATION RESULTS (Temporal IoU >= {args.iou})")
    print("=" * 65)
    header = f"{'Class':<20} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Prec':<6} | {'Recall':<6} | {'F1':<6}"
    print(header)
    print("-" * 65)

    for cls, metrics in sorted(results.items()):
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        prec = metrics["precision"]
        rec = metrics["recall"]
        f1 = metrics["f1"]
        print(f"{cls:<20} | {tp:<4} | {fp:<4} | {fn:<4} | {prec:<6.3f} | {rec:<6.3f} | {f1:<6.3f}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
