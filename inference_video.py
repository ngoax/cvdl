import argparse
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from model import ImprovedCNN
from gradcam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("best-weights.pt", map_location=device))
model.eval()
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

transform = A.Compose([
    A.Resize(64, 64),
    A.Normalize(),
    ToTensorV2()
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
target_layer = model.features[6]
gradcam = GradCAM(model, target_layer)

def overlay_heatmap(image, mask, alpha=0.3):
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    mask = np.clip(mask, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.float32(image) / 255 + alpha * (np.float32(heatmap) / 255)
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            pad = 0.1  
            x1 = max(int(x - pad * w), 0)
            y1 = max(int(y - pad * h), 0)
            x2 = min(int(x + w + pad * w), frame.shape[1])
            y2 = min(int(y + h + pad * h), frame.shape[0])
            w_ext = x2 - x1
            h_ext = y2 - y1

            face_roi = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            augmented = transform(image=face_rgb)
            input_tensor = augmented['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                pred = torch.argmax(logits, dim=1).item()
                probs = torch.softmax(logits, dim=1)[0]

            mask, _ = gradcam.generate(input_tensor, target_class=pred)
            mask = cv2.resize(mask, (w_ext, h_ext))
            mask = np.clip(mask, 0, 1)
            face_rgb = np.uint8(face_rgb)
            overlayed = overlay_heatmap(face_rgb, mask)
            overlayed = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)

            frame[y1:y2, x1:x2] = overlayed

            cv2.putText(frame, class_names[pred], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="output.mp4", help="Path to save output video")
    args = parser.parse_args()

    process_video(args.input, args.output)
