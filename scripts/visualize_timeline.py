"""
Timeline visualizer for predicted vs ground truth activity segments.

Produces a publication-ready horizontal Gantt-style timeline chart comparing
ground truth annotations against model predictions, colored by activity class.

Usage:
  python scripts/visualize_timeline.py --pred outputs/bench.png --gt data/ground_truth/gt.json --out outputs/plot.png
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


COLOR_MAP = {
    "Loitering": "#e74c3c",
    "Intrusion": "#e67e22",
    "Running": "#3498db",
    "Crowd Formation": "#9b59b6",
    "Abandoned Object": "#1abc9c",
    "Fire / Smoke": "#d35400",
    "Falling": "#c0392b",
}
DEFAULT_COLOR = "#7f8c8d"


def plot_timeline(pred_file: str, gt_file: str, output_path: str, title: str = "Activity Segments: Ground Truth vs Predictions"):
    with open(pred_file, "r") as f:
        p_data = json.load(f)
        preds = p_data.get("segments", p_data)

    with open(gt_file, "r") as f:
        g_data = json.load(f)
        gt = g_data.get("segments", g_data)

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=200)

    present_classes = set()

    for seg in gt:
        start = seg["start"]
        duration = max(0.5, seg["end"] - start)
        label = seg["label"]
        color = COLOR_MAP.get(label, DEFAULT_COLOR)
        present_classes.add(label)
        ax.broken_barh([(start, duration)], (0.8, 0.4), facecolors=color, edgecolor="black", alpha=0.9, linewidth=1)

    for seg in preds:
        start = seg["start"]
        duration = max(0.5, seg["end"] - start)
        label = seg["label"]
        color = COLOR_MAP.get(label, DEFAULT_COLOR)
        present_classes.add(label)
        ax.broken_barh([(start, duration)], (0.2, 0.4), facecolors=color, edgecolor="black", alpha=0.75, linewidth=0.8)

    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.4, 1.0])
    ax.set_yticklabels(["Predicted Segments", "Ground Truth"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    patches = [
        mpatches.Patch(color=COLOR_MAP.get(cls, DEFAULT_COLOR), label=cls)
        for cls in sorted(present_classes)
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved timeline plot to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", default="outputs/timeline.png")
    parser.add_argument("--title", default="Activity Segments: Ground Truth vs Predictions")
    args = parser.parse_args()
    plot_timeline(args.pred, args.gt, args.out, args.title)
