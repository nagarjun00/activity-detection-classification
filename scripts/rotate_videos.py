import cv2
import os
import subprocess
import json

def rotate_and_resize(input_path, output_path, target_w=1280, target_h=720):
    cmd = [".\\ffmpeg.bat", "-i", input_path,
           "-vf", "transpose=1,scale=%d:%d" % (target_w, target_h),
           "-y", output_path]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return os.path.exists(output_path)

print("Rotating hospital_waiting_cctv.mp4...")
rotate_and_resize("data/uploads/hospital_waiting_cctv.mp4",
                   "data/uploads/hospital_waiting_cctv_rotated.mp4")
if os.path.exists("data/uploads/hospital_waiting_cctv_rotated.mp4"):
    cap = cv2.VideoCapture("data/uploads/hospital_waiting_cctv_rotated.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print("  -> 720x%d" % h)

print("Rotating intrusion_scene.mp4...")
rotate_and_resize("data/uploads/intrusion_scene.mp4",
                   "data/uploads/intrusion_scene_rotated.mp4")
if os.path.exists("data/uploads/intrusion_scene_rotated.mp4"):
    cap = cv2.VideoCapture("data/uploads/intrusion_scene_rotated.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print("  -> 720x%d" % h)

# Replace originals with rotated versions
if os.path.exists("data/uploads/hospital_waiting_cctv_rotated.mp4"):
    os.replace("data/uploads/hospital_waiting_cctv_rotated.mp4",
               "data/uploads/hospital_waiting_cctv.mp4")
    print("Replaced hospital_waiting_cctv.mp4")

if os.path.exists("data/uploads/intrusion_scene_rotated.mp4"):
    os.replace("data/uploads/intrusion_scene_rotated.mp4",
               "data/uploads/intrusion_scene.mp4")
    print("Replaced intrusion_scene.mp4")

# Clean up intruder_cctv.mp4 (not needed anymore)
if os.path.exists("data/uploads/intruder_cctv.mp4"):
    os.remove("data/uploads/intruder_cctv.mp4")
    print("Removed intruder_cctv.mp4 (trimmed version kept)")

print("\nDone. Final videos:")
for f in sorted(os.listdir('data/uploads')):
    p = os.path.join('data/uploads', f)
    if os.path.isfile(p) and f.endswith('.mp4'):
        cap = cv2.VideoCapture(p)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print("  %s: %dx%d" % (f, w, h))
