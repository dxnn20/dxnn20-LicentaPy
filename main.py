from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torch
from DermnetDataset import DermnetDataset  # Ensure this file defines a class named DermnetDataset with a 'classes' attribute.
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import copy

BATCH_SIZE = 64
IMAGE_SIZE = 300
patience = 7
early_stop_counter = 0

train_path = "archive/train"  # Folder containing your training images
test_path = "archive/test"    # Folder containing your test images

# Data transforms (same as used during training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),            # Augmentation: rotation
    transforms.RandomHorizontalFlip(),        # Augmentation: horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Augmentation: color jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
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

# Define model using transfer learning with ResNet-50-wide
model = models.wide_resnet50_2(pretrained=True)
num_ftrs = model.fc.in_features  # Access the 'fc' layer's input features
num_classes = len(full_dataset.classes)  # Assumes DermnetDataset sets self.classes properly

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

# Define loss, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
# Reinitialize optimizer to ensure consistency if you change it:
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

# Set device and move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

# Set up TensorBoard writer
writer = SummaryWriter(log_dir='runs/experiment_resNet')

num_epochs = 50
best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
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
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    # Optionally clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
        writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

        # Save best model weights if validation loss improves
        if phase == 'val':
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1

    # Adjust learning rate based on the latest validation loss
    scheduler.step(epoch_loss)

    # After scheduler.step(epoch_loss):
    if early_stop_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

# Load the best model weights
model.load_state_dict(best_model_wts)
print(f"Best validation Loss: {best_val_loss:.4f}")
writer.close()

# Save the best model weights to disk for later use
torch.save(best_model_wts, 'best_model_weights_resnet50_wide.pth')
print("Model weights saved to 'best_model_weights_resnet50_wide.pth'")
