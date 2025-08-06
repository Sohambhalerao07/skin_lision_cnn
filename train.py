import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.preprocessing import SkinDataset, get_transforms, load_dataset
from models.model import get_model
from tqdm import tqdm

# Config
img_size = 224
batch_size = 32
epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metadata_csv = "data/train/HAM10000_metadata.csv"
img_dir = "data/train/HAM10000_images"

# Load data
df, class_names = load_dataset(metadata_csv, img_dir)
transform = get_transforms(img_size)
dataset = SkinDataset(df, img_dir, transform)

# Split
train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=df['dx'], random_state=42)
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

# Dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Model
model = get_model(len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# Save final model
torch.save(model.state_dict(), "skin_disease_classifier.pth")
print("✅ Model training complete and saved to 'skin_disease_classifier.pth'")
   