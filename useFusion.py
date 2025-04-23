import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from ResNet152Model import  ResNet152Model

# Import pytorch-grad-cam components
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from FusionModelClass import FusionModel

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load class names
train_path = "archive/train"
class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
num_classes = len(class_names)

print("Class names:")
print(class_names)

# Initialize and load the fusion model
model = ResNet152Model(num_classes)
model.load_state_dict(torch.load("best_resnet152_model.pth", map_location=device))

model = model.to(device)
model.eval()

# For Grad-CAM, we'll use the last layer of ResNet50 component
target_layers = [model.resnet.layer4[-1]]


# Create GradCAM object
cam = GradCAM(model=model, target_layers=target_layers)

# Load your test image
img_path = "archive/train/Bullous Disease Photos/benign-familial-chronic-pemphigus-1.jpg"
orig_image = Image.open(img_path).convert("RGB")
rgb_img = np.array(orig_image.resize((224, 224))) / 255.0  # Normalize to 0-1
input_tensor = data_transforms(orig_image).unsqueeze(0).to(device)

print("Processing image...")

# Get model prediction and confidence
# For inference only (not for Grad-CAM)
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    confidence_pct = confidence.item() * 100

predicted_class = class_names[predicted.item()]

# Create target for GradCAM
targets = [ClassifierOutputTarget(predicted.item())]

# For Grad-CAM (requires gradients)
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# Generate heatmap
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

# Show top 5 predictions
top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
top5_pct = [prob.item() * 100 for prob in top5_prob[0]]

print("\nTop 5 predictions:")
for i in range(5):
    class_idx = top5_indices[0][i].item()
    print(f"{class_names[class_idx]}: {top5_pct[i]:.2f}%")
