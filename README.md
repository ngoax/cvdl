# Facial Emotion Recognition with PyTorch

**Group members:** Antonia Härle, Clara Sophie Negwer, Andre Ngo, Jonas Saathoff

Project for Software Development Practical: Computer Vision and Deep Learning. It demonstrates (real-time) facial emotion recognition using a custom CNN architecture with ResNet-style blocks and Squeeze-and-Excitation (SE) channel attention.



## Overview

This project implements a facial emotion recognition model that can:
- Classify 6 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise
- Provide real-time webcam inference
- Generate Grad-CAM visualizations for Explainable AI
- Process video files with emotion predictions and heatmap overlays

## Project Structure

```
├── model.py                # Core CNN architecture
├── live_demo.py            # Real-time webcam demo
├── inference_video.py      # Video processing with Grad-CAM
├── gradcam.py              # Grad-CAM implementation
├── final_model_aug.ipynb   # Main training notebook
├── best-weights.pt         # Model weights
├── README.md               # This file
├── ferplus/                # FER+ dataset experiments
│   ├── ferplus.ipynb
│   ├── ferplus_all_aug.ipynb
│   └── best-weights-ferplus.pt
└── vary_augm_fer2013/      # Various Augmentation strategy experiments
    ├── no-aug.ipynb
    ├── horizontalflip-aug.ipynb
    ├── verticalflip_aug.ipynb
    └── mid-aug.ipynb
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Real-time Webcam Demo

Run the live emotion detection demo:

```bash
python3 live_demo.py
```

Press 'q' to quit the demo.

### Video Processing with Grad-CAM

Process a video file with emotion predictions and attention heatmaps:

```bash
python3 inference_video.py --input path/to/video.mp4 --output output_with_emotions.mp4
```

### Training

The main training pipeline is in [`final_model_aug.ipynb`](final_model_aug.ipynb). 


## Experimental Variations

The `vary_augm_fer2013/` directory contains experiments with different augmentation strategies:

1. **no-aug.ipynb**: Baseline without augmentation (~63% accuracy)
2. **horizontalflip-aug.ipynb**: Basic horizontal flipping (~66% accuracy)
3. **verticalflip_aug.ipynb**: Vertical flipping experiments (~62% accuracy)
4. **mid-aug.ipynb**: Moderate augmentation (~71% accuracy)

The `ferplus/` directory contains experiments on the FER+ dataset with refined labels.
