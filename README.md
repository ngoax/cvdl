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
├── model.py                        # Core CNN architecture
├── live_demo.py                    # Real-time webcam demo
├── inference_video.py              # Video processing with Grad-CAM
├── gradcam.py                      # Grad-CAM implementation
├── csv_gen.py                      # .csv generation file
├── final_model_ferplus.ipynb       # Main training notebook
├── best-weights.pt                 # Model weights
├── README.md                       # This file
├── experiments_ferplus/                        
│   ├── ferplus_moderate_aug.ipynb  # FER+ experiment moderate augmentation
└── experiments_fer2013/            # Various Augmentation strategy 
    ├── final_model_fer2013.ipynb   # Final model FER-2013
    ├── no_aug.ipynb
    ├── horizontalflip.ipynb
    ├── verticalflip.ipynb
    └── moderate_aug.ipynb
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

### .csv file generation

Iterates through images in a folder and outputs the corresponding classification scores in a csv file

```bash
python3 csv_gen.py path/to/imagefolder --ouptput predictions.csv
```


### Training

The main training pipeline is in [`final_model_ferplus.ipynb`](final_model_ferplus.ipynb). 


## Experimental Variations

The `experiments_fer2013/` directory contains experiments with different augmentation strategies:

1. **no_aug.ipynb**: Baseline without augmentation (~60% accuracy)
2. **horizontalflip.ipynb**: Basic horizontal flipping (~62% accuracy)
3. **verticalflip.ipynb**: Vertical flipping experiments (~57% accuracy)
4. **moderate_aug.ipynb**: Moderate augmentation (~71% accuracy)

The `experiments_ferplus/` directory contains experiment on the FER+ dataset with moderate augmentation strategy.
