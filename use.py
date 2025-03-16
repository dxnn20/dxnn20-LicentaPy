import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn

from GradCam import GradCAM  # Make sure this module is correctly implemented

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations (must match training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define the model using DenseNet121
num_classes = 23  # update as needed
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
# Replace the classifier with a Sequential block matching your training setup
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),  # Dropout with 20% probability
    nn.Linear(num_ftrs, num_classes)
)
# Load the saved model weights
model.load_state_dict(torch.load("best_model_weights_denseNet.pth", map_location=device))
model = model.to(device)
model.eval()

# For DenseNet121, a common choice is to use the last layer of the features module.
target_layer = model.features[-1]

# Initialize GradCAM with model and chosen layer
grad_cam = GradCAM(model, target_layer)

# Load your personal test image
img_path = "Screenshot 2025-03-10 114615.png"
orig_image = Image.open(img_path).convert("RGB")
input_tensor = data_transforms(orig_image)
input_tensor = input_tensor.unsqueeze(0).to(device)

# Generate Grad-CAM heatmap
cam = grad_cam.generate(input_tensor)

# Convert original image to numpy array for visualization
orig_np = np.array(orig_image.resize((224, 224)))
# Convert RGB to BGR for OpenCV
orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

# Apply a colormap to the heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
# Combine heatmap with the original image
overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

# Display the images using matplotlib
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(orig_np, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Overlay")
plt.axis("off")
plt.show()

# Optionally, get the predicted class:
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

# Automatically extract class names by scanning the training directory
train_path = "archive/train"
class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
print(f"Number of directories: {len(class_names)}")
print(f"Predicted class: {class_names[predicted.item()]}")
