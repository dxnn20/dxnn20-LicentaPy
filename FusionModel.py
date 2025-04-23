import torch
from FusionModelClass import FusionModel, HybridLoss
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from DermnetDataset import DermnetDataset


# Training configuration
BATCH_SIZE = 64
IMAGE_SIZE = 224
NUM_EPOCHS = 50
PATIENCE = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Enhanced data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset = DermnetDataset("useless/train", transform=train_transforms)
dataset_size = len(dataset)
num_classes = len(dataset.classes)

# Calculate class weights
class_counts = [0] * num_classes
for _, label in dataset:
    class_counts[label] += 1

total_samples = sum(class_counts)
class_weights = [total_samples / (num_classes * count) for count in class_counts]
class_weights = torch.FloatTensor(class_weights).to(device)

# Train-validation split
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = FusionModel(num_classes)
model = model.to(device)

# Loss function with class weights
criterion = HybridLoss(alpha=0.75, weight=class_weights, label_smoothing=0.1)

# Optimizer with different learning rates
params = [
    {'params': model.efficientnet_b0.parameters(), 'lr': 1e-5},
    {'params': model.resnet50.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
]

optimizer = optim.AdamW(params, weight_decay=0.01)

# Learning rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=[1e-4, 1e-4, 5e-4],
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.3
)

# Training loop
best_val_loss = float("inf")
early_stop_counter = 0

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
                    scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == 'val' and epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_fusion_model.pth")
        elif phase == 'val':
            early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training complete. Model saved.")
