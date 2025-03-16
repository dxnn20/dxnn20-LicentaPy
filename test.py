import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import os

from DermnetDataset import DermnetDataset

# Define the model using DenseNet121
num_classes = 23
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),  # Dropout with 20% probability
    nn.Linear(num_ftrs, num_classes)
)
model.load_state_dict(torch.load("best_model_weights_denseNet.pth", map_location=torch.device("cpu")))

# Define data transformations (must be the same as used during training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Path to test dataset
test_path = "archive/test"

# Create the test dataset and dataloader
test_dataset = DermnetDataset(test_path, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Set the model to evaluation mode
model.eval()

# If using GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Variables to track performance
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # Get the predicted class (index with highest logit)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
