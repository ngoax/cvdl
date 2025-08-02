# Facial Emotion Recognition with PyTorch

**Group members:** Antonia Härle, Clara Sophie Negwer, Andre Ngo, Jonas Saathoff

Project for Software Development Practical: Compputer Vision and Deep Learning. It demonstrates (real-time) facial emotion recognition using a custom CNN architecture with ResNet-style blocks and Squeeze-and-Excitation (SE) channel attention.



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
pip install torch torchvision
pip install opencv-python
pip install albumentations
pip install matplotlib seaborn
pip install scikit-learn
pip install tqdm
pip install torchcam
```

## Usage

### Real-time Webcam Demo

Run the live emotion detection demo:

```bash
python live_demo.py
```

Press 'q' to quit the demo.

### Video Processing with Grad-CAM

Process a video file with emotion predictions and attention heatmaps:

```bash
python inference_video.py --input path/to/video.mp4 --output output_with_emotions.mp4
```

### Training

The main training pipeline is in [`final_model_aug.ipynb`](final_model_aug.ipynb). Key features:
- Data augmentation with Albumentations
- Class-balanced loss weighting
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics

### Model Interpretability

Generate Grad-CAM visualizations using the [`GradCAM`](gradcam.py) class:

```python
from gradcam import GradCAM, show_gradcam_on_image
from model import ImprovedCNN

model = ImprovedCNN(num_classes=6)
model.load_state_dict(torch.load("best-weights.pt"))
target_layer = model.features[6]  # Target layer for visualization
gradcam = GradCAM(model, target_layer)
```

## Dataset

The project uses the FER2013 dataset with 6 emotion classes:
- **Training**: ~28K images
- **Validation**: ~6K images
- **Image size**: 64x64 pixels (resized)
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise

### Data Augmentation

Aggressive augmentation strategy for better generalization:
- Horizontal flipping (50%)
- Random brightness/contrast (30%)
- Shift/scale/rotation (70%)
- Gaussian blur (10%)
- Coarse dropout (25%)
- Normalization with dataset-specific statistics


## Experimental Variations

The `vary_augm_fer2013/` directory contains experiments with different augmentation strategies:

1. **no-aug.ipynb**: Baseline without augmentation (~63% accuracy)
2. **horizontalflip-aug.ipynb**: Basic horizontal flipping (~66% accuracy)
3. **verticalflip_aug.ipynb**: Vertical flipping experiments (~62% accuracy)
4. **mid-aug.ipynb**: Moderate augmentation (~71% accuracy)

The `ferplus/` directory contains experiments on the FER+ dataset with refined labels.
