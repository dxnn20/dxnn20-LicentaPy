import tensorflow as tf
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torch
from DermnetDataset import DermnetDataset
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import copy

# Confirm what hardware TF is using (for info only)
gpu_devices = tf.config.list_physical_devices('GPU')
if not gpu_devices:
    print("TensorFlow is using the CPU.")
else:
    print(f"TensorFlow is using the following GPU(s): {gpu_devices}")

BATCH_SIZE = 64
IMAGE_SIZE = 300

train_path = "archive/train"  # Folder containing your training images
test_path = "archive/test"  # Folder containing your test images

# Data transforms (same as used during training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Create full dataset from training folder
full_dataset = DermnetDataset(train_path, transform=data_transforms)
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Split the dataset into training and validation sets
dataset_train, dataset_val = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)

# Define model using transfer learning with ResNet-50
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = len(full_dataset.classes)  # assumes your DermnetDataset sets self.classes
model.fc = nn.Linear(num_ftrs, num_classes)

# Define loss, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

writer = SummaryWriter(log_dir='runs/experiment_1')

num_epochs = 2
best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print('-' * 10)

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
                    # Optionally, clip gradients
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
        if phase == 'val' and epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Adjust the learning rate based on validation loss
    scheduler.step(epoch_loss)

# Load the best model weights
model.load_state_dict(best_model_wts)
print(f"Best validation Loss: {best_val_loss:.4f}")
writer.close()

torch.save(best_model_wts, 'best_model_weights.pth')
