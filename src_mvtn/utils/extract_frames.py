import cv2
import os

# NOTE: Make sure OpenCV is installed: pip install opencv-python
# REFERNCE: https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/

# Input and output folders - [change these paths to your directories]
input_folder = r"C:\Users\Althea\COLLEGE\THESIS\Dataset\FSL-105 A dataset for recognizing 105 Filipino sign language videos\clips\1"
output_folder = r"C:\Users\Althea\COLLEGE\THESIS\MVTN\src_mvtn\datasets\FSL105_Frames\1"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Number of frames to extract from each video
num_frames = 40

# Loop through all X number of videos (0.mov to n.mov)
# number depends on how many videos each folder has
for vid_num in range(21):
    video_path = os.path.join(input_folder, f"{vid_num}.mov")
    save_dir = os.path.join(output_folder, str(vid_num))

    # Make a subfolder for each video
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick evenly spaced frame indices
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    count, extracted = 0, 0
    success = True

    while success and extracted < num_frames:
        success, frame = cap.read()
        if not success:
            break

        if count in frame_indices:
            filename = os.path.join(save_dir, f"frame_{extracted+1}.jpg")
            cv2.imwrite(filename, frame)
            extracted += 1

        count += 1

    cap.release()
    print(f"Extracted {extracted} frames from {vid_num}.mp4 into {save_dir}")
