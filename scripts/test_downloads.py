import sys, os, json, cv2
sys.path.insert(0, '.')

from app.pipeline.detect_track import run_detection_tracking
from app.pipeline.analyzers.loitering import detect_loitering
from app.pipeline.analyzers.intrusion import detect_intrusion
from app.pipeline.analyzer_runner import run_selected_analyzers
from app.pipeline.aggregator import segments_to_dict
from app.eval.temporal_iou import evaluate_per_class

ZONE_CENTER = [[420, 200], [860, 200], [860, 650], [420, 650]]
ZONE_RESTRICTED = [[500, 250], [780, 250], [780, 500], [500, 500]]

tests = [
    ("data/uploads/hospital_waiting_cctv.mp4",
     "data/ground_truth/loitering_gt.json",
     ["loitering"], ZONE_CENTER, "Loitering (Hospital)"),
    ("data/uploads/intrusion_scene.mp4",
     "data/ground_truth/intrusion_gt.json",
     ["intrusion"], ZONE_RESTRICTED, "Intrusion (Corridor)"),
]

for video_path, gt_path, activities, zone, label in tests:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frames / fps if fps else 0
    cap.release()
    print("%dx%d @ %s, %.1fs" % (w, h, fps, dur))

    print("Running detection...")
    det_frames = run_detection_tracking(video_path)
    unique_tracks = set(f.track_id for f in det_frames)
    cls_counts = {}
    for f in det_frames:
        cls_counts[f.cls] = cls_counts.get(f.cls, 0) + 1
    print("  Tracks: %d, Classes: %s, Total frames: %d" % (len(unique_tracks), cls_counts, len(det_frames)))

    # Show first few track positions
    if det_frames:
        sample = det_frames[:5]
        for sf in sample:
            bbox = sf.bbox
            print("  Track %d: cls=%s, bbox=%s, ts=%.2f" % (sf.track_id, sf.cls, bbox, sf.timestamp))

    segments = run_selected_analyzers(det_frames, video_path, zone, activities)
    predictions = segments_to_dict(segments)

    with open(gt_path) as f:
        gt = json.load(f).get("segments", [])

    results = evaluate_per_class(predictions, gt, iou_threshold=0.3)

    print("Predictions (%d segments):" % len(predictions))
    for p in predictions:
        print("  %s: %.1fs-%.1fs (conf: %.2f, tracks: %s)" % (
            p['label'], p['start'], p['end'], p['confidence'], p['track_ids'][:5]))

    print("Ground Truth (%d segments):" % len(gt))
    for g in gt:
        print("  %s: %.1fs-%.1fs" % (g['label'], g['start'], g['end']))

    print("Evaluation:")
    for cls, metrics in sorted(results.items()):
        total = metrics['true_positives'] + metrics['false_positives'] + metrics['false_negatives']
        if total > 0:
            print("  %s: TP=%d, FP=%d, FN=%d, P=%.3f, R=%.3f, F1=%.3f" % (
                cls, metrics['true_positives'], metrics['false_positives'],
                metrics['false_negatives'], metrics['precision'],
                metrics['recall'], metrics['f1']))

print("\nDone.")
