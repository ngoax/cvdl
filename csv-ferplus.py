import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import ImprovedCNN


EMOTION_CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

MEAN = (0.5072, 0.5072, 0.5072)
STD  = (0.2062, 0.2062, 0.2062)

weights_path = "best-weights-ferplus.pt"

def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = MEAN, std=STD)
    ])

def load_model(weights_path, device):
    model = ImprovedCNN(num_classes=len(EMOTION_CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_images(folder_path, model_path, output_csv="predictions.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    transform = get_transform()

    results = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

            results.append([img_path] + [f"{score:.2f}" for score in probs])

    df = pd.DataFrame(results, columns=["filepath"] + EMOTION_CLASSES)
    df.to_csv(output_csv, index=False, sep = ';')
    print(f" Saved CSV to: {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with input images")
    parser.add_argument("--model", default=weights_path, help="Path to model weights")
    parser.add_argument("--output", default="predictions.csv", help="CSV output path")
    args = parser.parse_args()

    predict_images(args.folder, args.model, args.output)
