from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Define custom dataset for Dermnet
class DermnetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.samples = []
        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, fname), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
