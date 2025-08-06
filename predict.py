import torch
from torchvision import transforms
from PIL import Image
import sys
import os

from models.model import get_model  # This should exist

# Config
image_path = "data/train/HAM10000_images/ISIC_0027419.jpg"  # <-- Change this
model_path = "skin_disease_classifier.pth"
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Predict function
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        return
    
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prediction = class_names[pred.item()]
    
    print(f"✅ Prediction for {os.path.basename(img_path)}: {prediction}")

# Run
if __name__ == "__main__":
    predict_image(image_path)
