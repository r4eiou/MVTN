# MVTN
MVTN: A Multiscale Video Transformer Network for Hand Gesture Recognition

```
Draft guide:
src_mvtn/
│
├── datasets/
│   ├── FSL105_ResizedFrames/         # Dataset: Resized frames (224x224) from FSL105 videos
│   │   ├── 0/                        # Sign 0 (class folder)
│   │   │   ├── 0/                    # Video 0 of sign 0
│   │   │   │   ├── frame_1.jpg
│   │   │   │   ├── frame_2.jpg
│   │   │   │   └── ...
│   │   │   ├── 1/                    # Video 1 of sign 0
│   │   │   └── ...
│   │   ├── 1/                        # Sign 1
│   │   └── ...
│   │
│   ├── splits/                       # JSONs that list train/val/test samples
│   │   ├── train.json                # List of training videos
│   │   ├── val.json                  # List of validation videos
│   │   ├── test.json                 # List of test videos
│   │   └── label.json                # Mapping of class indices (0–103) to labels
│
├── utils/                            # Data processing utilities
│   ├── normalize.py                  # Normalize frames (mean/std)
│   ├── normals.py                    # Surface normals (not used for RGB FSL105)
│   ├── optical_flow.py               # Optical flow extractor (not used for FSL105)
│   ├── read_data.py                  # Reads JSONs and frame folders into memory
│   ├── utils_briareo.py              # Briareo dataset utilities
│   ├── Briareo.py                    # Briareo dataset loader
│   ├── NVGestures.py                 # NVGestures dataset loader
│
├── hyperparameters/                  # Config files for training & testing
│   ├── Briareo/
│   │   ├── train.json
│   │   └── test.json
│   │
│   ├── NVGestures/
│   │   ├── train.json
│   │   └── test.json
│   │
│   └── (Create FSL105/ here)         # TODO: add configs for FSL105
│
├── Models/
│   ├── backbones/                    # Feature extractors
│   │   ├── resnet.py                 # ResNet backbone (frame-level feature extractor)
│   │   ├── c3d.py                    # C3D backbone
│   │   ├── r3d.py                    # ResNet3D backbone
│   │   └── vgg.py                    # VGG backbone
│   │
│   ├── attention.py                  # Multiscale attention implementation
│   ├── model_utilizer.py             # Helper for model initialization/loading
│   ├── module.py                     # Core modules of MVTN
│   ├── temporal.py                   # Temporal sequence encoding
│   └── test.py                       # Script for testing models
│
├── utils/ (root-level)               # General training utilities
│   ├── average_meter.py              # Tracks running averages (loss, accuracy)
│   ├── configer.py                   # Loads configs from hyperparameters/
│   ├── extract_frames.py             # Extracts & resizes frames (custom, for FSL105)
│   ├── generate_splits.py            # Generates train/val/test JSON splits (custom, for FSL105)
│   └── visualization.py              # Training progress visualization
│
├── cs.py / cs2.py                    # Custom experiment scripts
├── main.py                           # Main pipeline (loads configs, datasets, model)
├── train.py                          # Training entry point
├── test.py                           # Testing entry point
```
