import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from DermnetDataset import DermnetDataset  # Ensure this file defines a class named DermnetDataset with a 'classes' attribute.

BATCH_SIZE = 64
IMAGE_SIZE = 224
PATIENCE = 7
EARLY_STOP_COUNTER = 0
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

train_path = "archive/train"  # Folder containing your training images
test_path = "archive/test"    # Folder containing your test images

# Data transforms with advanced augmentation
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the full dataset from the training folder
full_dataset = DermnetDataset(train_path, transform=data_transforms)
dataset_size = len(full_dataset)
print(f"Total number of images in dataset: {dataset_size}")
print(f"Classes found: {full_dataset.classes}")

# Split the dataset into training and validation sets (80/20 split)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
dataset_train, dataset_val = random_split(full_dataset, [train_size, val_size])
print(f"Training images: {train_size}, Validation images: {val_size}")

# Create DataLoaders for training and validation
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

# Define model using transfer learning with ResNet-50
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = len(full_dataset.classes)

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

# Define loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)

# Set device and move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

# Set up TensorBoard writer
writer = SummaryWriter(log_dir='runs/experiment_resNet18')

# Initialize variables for tracking best model
best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')

# Initialize EMA weights
ema_model = copy.deepcopy(model)
ema_decay = 0.999

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print('*' * 40)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer.step()
                        scheduler.step()

                        # Update EMA weights
                        with torch.no_grad():
                            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
        writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

        # Track the best model
        if phase == 'val' and epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            EARLY_STOP_COUNTER = 0
        elif phase == 'val':
            EARLY_STOP_COUNTER += 1

    # Early stopping check
    if EARLY_STOP_COUNTER >= PATIENCE:
        print("Early stopping triggered.")
        break

# Load the best model weights
model.load_state_dict(best_model_wts)
print("Training complete.")

# Save the model
torch.save(model.state_dict(), "resnet18_best_model.pth")
print("Model saved.")
