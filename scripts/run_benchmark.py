"""Run a suite of labelled videos through only their relevant analyzers.

Usage:
  python scripts/run_benchmark.py
  python scripts/run_benchmark.py --manifest data/ground_truth/benchmark_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.eval.temporal_iou import evaluate_per_class
from app.pipeline.analyzer_runner import (
    TRACK_BASED_ACTIVITIES,
    run_selected_analyzers,
    validate_activities,
)
from app.pipeline.aggregator import segments_to_dict
from app.pipeline.detect_track import run_detection_tracking


def load_segments(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("segments", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark labelled activity videos.")
    parser.add_argument(
        "--manifest",
        default="data/ground_truth/benchmark_manifest.json",
        help="Benchmark manifest JSON path, relative to repository root.",
    )
    parser.add_argument("--iou", type=float, default=0.3, help="Temporal IoU threshold.")
    parser.add_argument(
        "--output-dir", default="outputs/benchmark", help="Directory for predictions and summary."
    )
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: List[Dict] = []
    all_ground_truth: List[Dict] = []
    video_results = []

    for entry in manifest["videos"]:
        name = entry["name"]
        video_path = REPO_ROOT / entry["video"]
        ground_truth_path = REPO_ROOT / entry["ground_truth"]
        activities = validate_activities(entry["activities"])
        zone = [tuple(point) for point in entry.get("zone", [(0, 0), (1920, 0), (1920, 1080), (0, 1080)])]

        if not video_path.exists() or not ground_truth_path.exists():
            raise FileNotFoundError(f"Missing benchmark input for '{name}'")

        print(f"\nBenchmarking {name}: {', '.join(sorted(activities))}")
        frames = run_detection_tracking(str(video_path)) if activities & TRACK_BASED_ACTIVITIES else []
        segments = run_selected_analyzers(frames, str(video_path), zone, activities)
        predictions = segments_to_dict(segments)
        ground_truth = load_segments(ground_truth_path)
        metrics = evaluate_per_class(predictions, ground_truth, args.iou)

        with (output_dir / f"{name}_predictions.json").open("w", encoding="utf-8") as file:
            json.dump({"video": video_path.name, "activities": sorted(activities), "segments": predictions}, file, indent=2)

        print(json.dumps(metrics, indent=2))
        video_results.append({"name": name, "metrics": metrics})
        all_predictions.extend(predictions)
        all_ground_truth.extend(ground_truth)

    summary = {
        "iou_threshold": args.iou,
        "videos": video_results,
        "overall_metrics": evaluate_per_class(all_predictions, all_ground_truth, args.iou),
    }
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nOVERALL METRICS")
    print(json.dumps(summary["overall_metrics"], indent=2))
    print(f"\nSaved benchmark results to {output_dir}")


if __name__ == "__main__":
    main()
