import cv2
import os
import numpy as np

# Base input and output paths
base_input = r"C:\Users\Althea\COLLEGE\THESIS\Dataset\FSL-105 A dataset for recognizing 105 Filipino sign language videos\clips"
base_output = r"C:\Users\Althea\COLLEGE\THESIS\MVTN\src_mvtn\datasets\FSL105_Frames"

# Number of frames to extract from each video
num_frames = 40

# not sure
# frame_size = (224, 224)

def numeric_sort(files):
    """Sort file list numerically by filename before extension."""
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))

# Loop through each subfolder (sorted: 0, 1, ..., 103)
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

        # Pick evenly spaced frame indices without duplicates
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        count, extracted = 0, 0
        success = True

        while success and extracted < num_frames:
            success, frame = cap.read()
            if not success:
                break

            if count in frame_indices:
                # resized_frame = cv2.resize(frame, frame_size)
                filename = os.path.join(save_dir, f"frame_{extracted+1}.jpg")
                cv2.imwrite(filename, frame) 
                extracted += 1

            count += 1

        cap.release()
        print(f"  Extracted {extracted} frames from {video_file} into {save_dir}")
