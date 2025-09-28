import os
import json
import random

# Root folder where videos are stored per sign
root = r"C:\Users\Althea\COLLEGE\THESIS\MVTN\src_mvtn\datasets\FSL105"
output_dir = os.path.join(root, "splits")  # folder to save JSON files

os.makedirs(output_dir, exist_ok=True)

# Splits ratios
train_ratio, val_ratio = 0.7, 0.15
splits = {"train": [], "val": [], "test": []}

# get the list of sign directories; should be numeric (0-103)
signs = [d for d in os.listdir(root) if d.isdigit()]
signs = sorted(signs, key=lambda x: int(x))

# loop through each sign directory to gather videos
for sign in signs:
    sign_path = os.path.join(root, sign)
    videos = [v for v in os.listdir(sign_path) if os.path.isfile(os.path.join(sign_path, v))]
    print(f"Sign {sign} videos found:", videos)  # debugging print
    random.shuffle(videos)

    n = len(videos)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    for i, vid in enumerate(videos):
        video_path = os.path.join(sign_path, vid)
        # Replace backslashes with forward slashes for consistency
        video_path = video_path.replace("\\", "/")

        entry = {
            "video_id": f"sign{sign}_vid{vid.split('.')[0]}",
            "video_path": video_path,
            "label": int(sign)
        }

        if i < n_train:
            splits["train"].append(entry)
        elif i < n_train + n_val:
            splits["val"].append(entry)
        else:
            splits["test"].append(entry)

# Save JSON files
for split in ["train", "val", "test"]:
    with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
        json.dump(splits[split], f, indent=2)

# Save label map (0–103)
label_map = {int(sign): int(sign) for sign in signs}
with open(os.path.join(output_dir, "labels.json"), "w") as f:
    json.dump(label_map, f, indent=2)

print("Done! JSON splits saved in", output_dir)