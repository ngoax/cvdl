import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import ImprovedCNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedCNN().to(device)
try:
    model.load_state_dict(torch.load("best-weights.pt", map_location=device))
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("Error: 'best-weights.pt' file not found.")
    exit(1)
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

model.eval()
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

transform = A.Compose([
    A.Resize(64, 64),
    A.Normalize(),
    ToTensorV2()
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Press 'q' to quit demo.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        augmented = transform(image=face_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(logits, dim=1).item()

        label = class_names[pred]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        prob_y_start = y + h + 20
        for i, (emotion, prob) in enumerate(zip(class_names, probabilities)):
            prob_text = f'{emotion}: {prob.item():.2f}'
            prob_y = prob_y_start + i * 25
            color = (0, 255, 0) if i == pred else (255, 255, 255)
            cv2.putText(frame, prob_text, (x, prob_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imshow('Webcam Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
