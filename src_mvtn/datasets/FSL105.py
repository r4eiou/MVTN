import os
import math
import torch
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from datasets.utils.normalize import normalize
import json

class FSL105(Dataset):
    """FSL105 Dataset class - Briareo-style, RGB frames only"""
    def __init__(self, configer, path, split="train", data_type='rgb', transforms=None, n_frames=40, optical_flow=False):
        """Constructor method for FSL105 Dataset class
        
        Args:
            configer: Configer object
            path: dataset root path
            split: train/val/test
            data_type: RGB only for FSL105 (kept for Briareo signature)
            transforms: optional transforms
            n_frames: number of frames per clip
            optical_flow: ignored for FSL105
        """
        super().__init__()

        self.dataset_path = Path(path)
        self.split = split
        self.data_type = data_type          # kept for signature
        self.transforms = transforms
        self.n_frames = n_frames
        self.optical_flow = optical_flow    # kept for signature

        print(f"Loading FSL105 {split.upper()} dataset...", end=" ")

        # Load JSON split
        split_file = self.dataset_path / "splits" / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"{split_file} does not exist!")
        with open(split_file, "r") as f:
            data = json.load(f)

        # Prepare clips with exactly n_frames
        fixed_data = []
        for record in data:
            # Directly keep the folder path string
            record["frames"] = record["frames"]
            fixed_data.append(record)

        self.data = fixed_data
        print("done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = sample["label"]

        frames_folder = Path(sample["frames"])
        if not frames_folder.is_absolute():
            frames_folder = (self.dataset_path / frames_folder).resolve()
        else:
            frames_folder = frames_folder.resolve()
        # print(f"[DEBUG] Reading frames from folder: {frames_folder}") # Debug print

        # Read all JPG frames from the folder
        frame_files = sorted(frames_folder.glob("*.jpg"))
        # print(f"[DEBUG] Found {len(frame_files)} frames. Sample files: {frame_files[:5]}") # Debug print

        if len(frame_files) < self.n_frames:
            raise ValueError(f"Not enough frames in {frames_folder}")

        # Sample n_frames evenly
        if len(frame_files) > self.n_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.n_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        else:
            frame_files = frame_files[:self.n_frames]

        # Load frames
        clip = []
        for f in frame_files:
            img = cv2.imread(str(f))
            if img is None:
                raise FileNotFoundError(f"Cannot read image {f}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize immediately to fixed size
            img = cv2.resize(img, (224, 224))
            clip.append(img)

        clip = np.array(clip)  # (T, H, W, C)

        # print(f"[DEBUG] Clip shape after resizing: {clip.shape}, should be ({self.n_frames}, 224, 224, 3)")

        # Normalize
        clip = normalize(clip)

        # Apply augmentations
        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([cv2.resize(aug_det.augment_image(clip[i]), (224, 224)) for i in range(clip.shape[0])])

        # Convert to tensor in (C, T, H, W) like Briareo
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float()
        label = torch.tensor(label, dtype=torch.long)

        return clip, label