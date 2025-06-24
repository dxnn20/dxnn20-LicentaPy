# Install the library if you haven't already

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Import pytorch-grad-cam components
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define the model
num_classes = 23
model = models.resnet50(pretrained=False)  # Initialize the model first
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

# Load the saved model weights
model.load_state_dict(torch.load("best_fusion_model.pth", weights_only=True))
model = model.to(device)
model.eval()

# Define target layers - try different layers to see which gives better visualization
target_layers = [
    model.layer1[-1],  # Last bottleneck of layer1
    model.layer2[-1],  # Last bottleneck of layer2
    model.layer3[-1],  # Last bottleneck of layer3
    model.layer4[-1]   # Last bottleneck of layer4
]

for param in model.parameters():
    param.requires_grad = True

# Create GradCAM object
cam = GradCAM(model=model, target_layers=target_layers)

# Load your test image
img_path = "useless/test/Eczema Photos/Dyshidrosis-54.jpg"
orig_image = Image.open(img_path).convert("RGB")
rgb_img = np.array(orig_image.resize((224, 224))) / 255.0  # Normalize to 0-1
input_tensor = data_transforms(orig_image).unsqueeze(0).to(device)

print("Calculating predictions...")
outputs = model(input_tensor)
probabilities = F.softmax(outputs, dim=1)
confidence, predicted = torch.max(probabilities, 1)
confidence_pct = confidence.item() * 100

# Get class names
train_path = "useless/train"
class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
predicted_class = class_names[predicted.item()]

# Create target for GradCAM
targets = [ClassifierOutputTarget(predicted.item())]

# Generate heatmap
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]  # Get the first image in the batch

# Create visualization
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title(f"Original: {predicted_class}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(grayscale_cam, cmap='jet')
plt.title(f"Grad-CAM: {confidence_pct:.2f}% confidence")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(visualization)
plt.title("Overlay")
plt.axis("off")
plt.show()

# Print prediction with confidence
print(f"Predicted class: {predicted_class} with {confidence_pct:.2f}% confidence")

# Optionally show top 5 predictions
top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
top5_pct = [prob.item() * 100 for prob in top5_prob[0]]

print("\nTop 5 predictions:")
for i in range(5):
    class_idx = top5_indices[0][i].item()
    print(f"{class_names[class_idx]}: {top5_pct[i]:.2f}%")
