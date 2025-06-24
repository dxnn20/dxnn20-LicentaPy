import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

from sklearn.metrics import classification_report, confusion_matrix
import cv2
from tqdm import tqdm
from PIL import Image

PATIENCE = 15

CONFIG = {
    "image_size": (470, 470),
    "batch_size": 16,
    "num_epochs": 70,
    "num_workers": 2,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "data_dir": "/kaggle/input/final-derm/archive/train",
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "mixup_alpha": 0.4,
    "label_smoothing": 0.1
}

print(f"Using device: {CONFIG['device']}")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(CONFIG["image_size"][0], scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(CONFIG["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DermnetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []
        self.targets = []
        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, fname), label))
                self.targets.append(label)
        self.targets = torch.tensor(self.targets)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image.float(), label


dataset = DermnetDataset(CONFIG["data_dir"], transform=train_transforms)
dataset_size = len(dataset)
num_classes = len(dataset.classes)
print(f"Classes: {num_classes}, Dataset size: {dataset_size}")

train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"],
                        pin_memory=True)


class EfficientNetB5Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB5Model, self).__init__()
        self.efficientnet = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.efficientnet.classifier[1].in_features, 512),
            nn.GELU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


model = EfficientNetB5Model(num_classes).to(CONFIG["device"])

criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

# Separate base and classifier parameters
base_params = list(model.efficientnet.features.parameters())
classifier_params = list(model.efficientnet.classifier.parameters())

optimizer = optim.AdamW([
    {'params': base_params, 'lr': 0.5 * CONFIG["lr"]},
    {'params': classifier_params, 'lr': 1.5 * CONFIG["lr"]}
], weight_decay=CONFIG["weight_decay"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=1, eta_min=5e-4
)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(CONFIG["device"])
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


scaler = torch.amp.GradScaler('cuda')

best_val_acc = 0.0
early_stop_counter = 0

for epoch in range(CONFIG["num_epochs"]):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['num_epochs']}"):
        inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
        optimizer.zero_grad(set_to_none=True)

        if np.random.rand() < 0.1:
            inputs_mixed, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=CONFIG["mixup_alpha"])
            with torch.amp.autocast('cuda'):
                outputs = model(inputs_mixed)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)
        total += labels.size(0)

    scheduler.step()

    train_loss = running_loss / total
    train_acc = running_corrects.double() / total

    model.eval()
    val_loss = 0.0
    val_corrects = 0
    total_val = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])

            outputs = model(inputs)

            if epoch >= 10:
                flipped_inputs = torch.flip(inputs, dims=[3])
                flipped_outputs = model(flipped_inputs)
                outputs = (outputs + flipped_outputs) / 2.0

            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels)
            total_val += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= total_val
    val_acc = val_corrects.double() / total_val
    print(
        f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_efficientnet_b5_model.pth")
        print(f"New best model saved with accuracy: {best_val_acc:.4f}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete. Best model saved.")