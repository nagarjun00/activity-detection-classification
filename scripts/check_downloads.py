import cv2
import os
import glob

files = glob.glob("data/uploads/intrusion_candidate_*") + glob.glob("data/uploads/loitering_candidate.*")

for f in sorted(files):
    if not os.path.exists(f):
        continue
    try:
        cap = cv2.VideoCapture(f)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = frames / fps if fps else 0
        size = os.path.getsize(f) / (1024 * 1024)
        base = os.path.basename(f)[:50]
        print("%s: %dx%d @ %sfps, %.1fs, %.1f MB" % (base, w, h, fps, dur, size))
        cap.release()
    except Exception as e:
        print("Error reading %s: %s" % (f[:50], str(e)[:80]))
