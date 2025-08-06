# evaluate.py
import torch
from torch.utils.data import DataLoader
from torchvision import models
from utils.preprocessing import load_dataset, get_transforms
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
transform = get_transforms()
dataset, labels, class_names = load_dataset("data/train/HAM10000_images", transform)
_, val_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels, random_state=42)

val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=32, shuffle=False)

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("skin_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

print(f"Validation Accuracy: {correct / total:.4f}")
