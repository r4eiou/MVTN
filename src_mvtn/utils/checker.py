import os
import cv2

# ANSI escape code for red text
RED = "\033[91m"
RESET = "\033[0m"

base = r"C:\Users\Althea\COLLEGE\THESIS\MVTN\src_mvtn\datasets\FSL105_ResizedFrames"

for folder in sorted(os.listdir(base)):
    folder_path = os.path.join(base, folder)
    if not os.path.isdir(folder_path):
        continue

    videos = sorted(os.listdir(folder_path))
    for video in videos:
        video_path = os.path.join(folder_path, video)
        if not os.path.isdir(video_path):
            continue

        bad_frames = []
        for root, dirs, files in os.walk(video_path):
            for f in files:
                if f.lower().endswith(".jpg"):
                    path = os.path.join(root, f)
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    if (h, w) != (224, 224):
                        bad_frames.append((path, (h, w)))

        if bad_frames:
            print(f"Video {video} in SubFolder {folder}: Found {len(bad_frames)} bad frames")
            for path, size in bad_frames:
                print(f"  {RED}{path} {size}{RESET}")
        else:
            print(f"Video {video} in SubFolder {folder}: clear, no bad frames")
