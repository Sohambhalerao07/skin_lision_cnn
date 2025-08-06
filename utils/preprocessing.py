import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(df['dx'].unique()))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    image = Image.open(row['path']).convert("RGB")
    if self.transform:
        image = self.transform(image)
    dx_label = torch.tensor(row['dx_idx'])
    loc_label = torch.tensor(row['loc_idx'])
    return image, dx_label, loc_label


def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def load_dataset(metadata_path, img_dir):
    df = pd.read_csv(metadata_path)
    df['path'] = df['image_id'].apply(lambda x: os.path.join(img_dir, x + ".jpg"))

    dx_classes = sorted(df['dx'].unique())
    loc_classes = sorted(df['localization'].unique())

    dx_to_idx = {name: i for i, name in enumerate(dx_classes)}
    loc_to_idx = {name: i for i, name in enumerate(loc_classes)}

    df['dx_idx'] = df['dx'].map(dx_to_idx)
    df['loc_idx'] = df['localization'].map(loc_to_idx)

    return df, dx_classes, loc_classes
