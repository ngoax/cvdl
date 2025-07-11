
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch import nn
import torch.nn.functional as F




# insert residual block: https://medium.com/@neetu.sigger/a-comprehensive-guide-to-understanding-and-implementing-bottleneck-residual-blocks-6b420706f66b

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MySTNCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 32x32

            ResidualBlock(64, 128, stride=2),   # -> 16x16
            ResidualBlock(128, 256, stride=2),  # -> 8x8
            ResidualBlock(256, 512, stride=2),  # -> 4x4

            nn.AdaptiveAvgPool2d((1, 1))      # Global Average Pooling
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)







# load  trained model 
# mps for M1 MAC
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MySTNCNN().to(device)
try:
    model.load_state_dict(torch.load("best-weights.pt", map_location=device))
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("Error: 'best-weights.pt' file not found. Please ensure the model weights file exists in the project directory.")
    exit(1)
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Using randomly initialized weights instead.")

model.eval()
# List your class names here (should match your training classes)
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

transform = A.Compose([
    A.Resize(64, 64),
    A.Normalize(),
    ToTensorV2()
])

# face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer - use full_backward_hook to avoid deprecation warning
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Pool gradients across spatial dimensions
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps - ensure same device
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / torch.max(cam) if torch.max(cam) > 0 else cam
        
        return cam.detach().cpu().numpy()

# Initialize Grad-CAM (after model is loaded)
grad_cam = GradCAM(model, model.features[6].conv2) 






# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the demo.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting.")
        break

# Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Convert face to RGB for albumentations
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        augmented = transform(image=face_rgb)
        
        input_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)  # Move tensor to same device as model


        input_tensor.requires_grad_()

        '''
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]  # Apply softmax to get probabilities

            pred = torch.argmax(logits, dim=1).item()
        '''

        # Remove torch.no_grad() - this was preventing gradients
        model.train()  # Enable gradient computation
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(logits, dim=1).item()
        model.eval()  # Set back to eval mode

        # Generate saliency map
        cam = grad_cam.generate_cam(input_tensor, pred)
        
        # Resize CAM to face size
        cam_resized = cv2.resize(cam, (w, h))
        cam_normalized = (cam_resized * 255).astype(np.uint8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
        
        # Overlay heatmap on face region
        face_region = frame[y:y+h, x:x+w]
        overlay = cv2.addWeighted(face_region, 0.6, heatmap, 0.4, 0)
        frame[y:y+h, x:x+w] = overlay

        label = class_names[pred]
        
        # Draw bounding box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display emotion label above the bounding box
        cv2.putText(frame, f'{label}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display all probabilities as text overlay
        prob_y_start = y + h + 20
        for i, (emotion, prob) in enumerate(zip(class_names, probabilities)):
            prob_text = f'{emotion}: {prob.item():.2f}'
            prob_y = prob_y_start + i * 25
            
            # Color code: green for highest probability, white for others
            color = (0, 255, 0) if i == pred else (255, 255, 255)
            
            cv2.putText(frame, prob_text, (x, prob_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


    cv2.imshow('Webcam Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()