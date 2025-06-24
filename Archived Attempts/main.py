import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from DermnetDataset import DermnetDataset

BATCH_SIZE = 64
IMAGE_SIZE = 224
NUM_EPOCHS = 50
PATIENCE = 7
EARLY_STOP_COUNTER = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")
# Improved Data Augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.1, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Dataset
dataset = DermnetDataset("useless/train", transform=train_transforms)
dataset_size = len(dataset)
num_classes = len(dataset.classes)

train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load ResNet50 Model with Pretrained Weights
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Modify the Fully Connected Layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.BatchNorm1d(1024),
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# Loss Function with Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

# Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Training Loop
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    print("*" * 25)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == 'val' and epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            EARLY_STOP_COUNTER = 0
            torch.save(model.state_dict(), "best_model_resnet50_part2.pth")
        elif phase == 'val':
            EARLY_STOP_COUNTER += 1

    scheduler.step()

    if EARLY_STOP_COUNTER >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training complete. Model saved.")
