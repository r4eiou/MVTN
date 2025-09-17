import cv2
import os
import numpy as np

# NOTE: Make sure OpenCV is installed: pip install opencv-python
# REFERNCE: https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/

# Base input and output paths - [change these paths to your directories]
base_input = r"C:\Users\Althea\COLLEGE\THESIS\Dataset\FSL-105 A dataset for recognizing 105 Filipino sign language videos\clips"
base_output = r"C:\Users\Althea\COLLEGE\THESIS\MVTN\src_mvtn\datasets\FSL105_ResizedFrames"

# Number of frames to extract from each video
num_frames = 40

## not yet sure ##
# Just set desired frame size here, however, image might look stretched or squished
# frame_size = (224, 224)

# resize and center crop function to 224x224
# center crop is used to maintain aspect ratio so that image won't look stretched or squished
def resize_and_center_crop(frame, crop_size=224, resize_shorter=256):
    h, w, _ = frame.shape
    # Resize shorter side
    if h < w:
        new_h = resize_shorter
        new_w = int(w * (resize_shorter / h))
    else:
        new_w = resize_shorter
        new_h = int(h * (resize_shorter / w))
    resized = cv2.resize(frame, (new_w, new_h))
    # Center crop
    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2
    cropped = resized[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cropped

def numeric_sort(files):
    # Sort file list numerically by filename before extension.
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))

# Loop through each subfolder (sorted: 0, 1, ..., 104) no need to manually change input/output paths
for folder in sorted(os.listdir(base_input), key=lambda x: int(x) if x.isdigit() else x):
    input_folder = os.path.join(base_input, folder)
    if not os.path.isdir(input_folder):
        continue  # skip if not a folder

    # Build matching output folder
    output_folder = os.path.join(base_output, folder)
    os.makedirs(output_folder, exist_ok=True)

    # List all video files (case-insensitive, sorted)
    videos = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".mov", ".mp4", ".avi"))])
    videos = numeric_sort(videos)
    print(f"\nProcessing folder {folder} with {len(videos)} videos...")

    # Loop through each video file
    for video_file in videos:
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # e.g., "0", "1", etc.
        save_dir = os.path.join(output_folder, video_name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"  Warning: could not read {video_file}")
            cap.release()
            continue

        # not yet sure
        # Pick evenly spaced frame indices without duplicates
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        count, extracted = 0, 0
        success = True

        while success and extracted < num_frames:
            success, frame = cap.read()
            if not success:
                break

            if count in frame_indices:
                # not yet sure
                # resized_frame = cv2.resize(frame, frame_size)
                processed_frame = resize_and_center_crop(frame)
                filename = os.path.join(save_dir, f"frame_{extracted+1}.jpg")
                cv2.imwrite(filename, processed_frame)
                # cv2.imwrite(filename, frame) # no resize version
                extracted += 1

            count += 1

        cap.release()
        print(f"  Extracted {extracted} frames from {video_file} into {save_dir}")
