import cv2, os, json, subprocess

def safe(s):
    return s.encode('ascii', errors='replace').decode('ascii')

# Clean up old candidate files (non-video, non-main clips)
for f in list(os.listdir('data/uploads')):
    if 'candidate' in f and not f.endswith('.mp4'):
        path = os.path.join('data/uploads', f)
        if os.path.isfile(path):
            try:
                os.remove(path)
                print("Removed: " + safe(f))
            except:
                pass

# Remove f616 and f299 (wrong/corrupt videos)
for f in list(os.listdir('data/uploads')):
    if '.f616' in f or '.f299' in f or '.f251' in f:
        path = os.path.join('data/uploads', f)
        if os.path.isfile(path):
            try:
                os.remove(path)
                print("Removed: " + safe(f))
            except:
                pass

# Rename loitering
loitering = "data/uploads/loitering_candidate.f398.mp4"
if os.path.exists(loitering):
    os.rename(loitering, "data/uploads/hospital_waiting_cctv.mp4")
    print("Renamed loitering to hospital_waiting_cctv.mp4")

print("\nTrimming intrusion video...")
intrusion_in = "data/uploads/intruder_cctv.mp4"
intrusion_out = "data/uploads/intrusion_scene.mp4"

if os.path.exists(intrusion_in):
    cap = cv2.VideoCapture(intrusion_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps else 0
    cap.release()
    print("Intrusion duration: %.1fs" % dur)
    trim_end = min(20.0, dur)
    if trim_end > 5.0:
        cmd = [".\\ffmpeg.bat", "-i", intrusion_in, "-ss", "5",
               "-to", str(trim_end), "-y", intrusion_out]
        subprocess.run(cmd, capture_output=True, timeout=120)
        if os.path.exists(intrusion_out):
            print("Trimmed intrusion: %s (%.1fs)" % (intrusion_out, trim_end - 5.0))
            gt = {"video": "intrusion_scene.mp4",
                  "segments": [{"start": 5.0, "end": trim_end, "label": "Intrusion"}]}
            with open("data/ground_truth/intrusion_gt.json", "w") as f:
                json.dump(gt, f, indent=2)
            print("GT: %s" % gt["segments"])

print("\nChecking loitering video...")
loitering_in = "data/uploads/hospital_waiting_cctv.mp4"
if os.path.exists(loitering_in):
    cap = cv2.VideoCapture(loitering_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps else 0
    cap.release()
    print("Loitering duration: %.1fs" % dur)
    if dur >= 13.0:
        gt = {"video": "hospital_waiting_cctv.mp4",
              "segments": [{"start": 5.0, "end": 13.0, "label": "Loitering"}]}
        with open("data/ground_truth/loitering_gt.json", "w") as f:
            json.dump(gt, f, indent=2)
        print("GT: %s" % gt["segments"])

print("\nFinal uploads:")
for f in sorted(os.listdir('data/uploads')):
    p = os.path.join('data/uploads', f)
    if os.path.isfile(p):
        print("  %s: %.1f MB" % (safe(f), os.path.getsize(p) / (1024 * 1024)))

print("\nGround truth files:")
for f in sorted(os.listdir('data/ground_truth')):
    if f.endswith('.json') and 'candidate' not in f and 'benchmark' not in f:
        with open(os.path.join('data/ground_truth', f)) as fh:
            data = json.load(fh)
            print("  %s: %s" % (safe(f), data.get('segments', [])))
