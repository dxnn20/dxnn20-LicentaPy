import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from FusionModelClass import FusionModel
from DermnetDataset import DermnetDataset

# Path to test dataset
test_path = "useless/test"

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the test dataset and dataloader
test_dataset = DermnetDataset(test_path, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get class names
class_names = test_dataset.classes
num_classes = len(class_names)
print(f"Number of test classes: {len(class_names)}")
print(f"Number of test images: {len(test_dataset)}")

# Initialize the fusion model
model = FusionModel(num_classes)

# Load the saved model weights
model.load_state_dict(torch.load("best_fusion_model_20Epochs.pth", map_location=torch.device("cpu")))

# Set the model to evaluation mode
model.eval()

# If using GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Variables to track performance
all_preds = []
all_labels = []
all_confidences = []

# Evaluate the model
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)

        # Get predictions and confidences
        confidences, predictions = torch.max(probabilities, dim=1)

        # Store results
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

# Calculate overall accuracy
accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

# Generate classification report
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('fusion_model_confusion_matrix.png')
plt.show()

# Calculate per-class accuracy
correct_per_class = {}
total_per_class = {}
for true_label, pred_label in zip(all_labels, all_preds):
    true_class = class_names[true_label]
    total_per_class[true_class] = total_per_class.get(true_class, 0) + 1
    if true_label == pred_label:
        correct_per_class[true_class] = correct_per_class.get(true_class, 0) + 1

# Print per-class accuracy
print("\nPer-class Accuracy:")
for class_name in class_names:
    if class_name in total_per_class and total_per_class[class_name] > 0:
        class_acc = correct_per_class.get(class_name, 0) / total_per_class[class_name]
        print(
            f"{class_name}: {class_acc * 100:.2f}% ({correct_per_class.get(class_name, 0)}/{total_per_class[class_name]})")
    else:
        print(f"{class_name}: No test samples")

# Analyze confidence distribution
plt.figure(figsize=(10, 6))
plt.hist(all_confidences, bins=20, alpha=0.7)
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Confidence Distribution')
plt.grid(True, alpha=0.3)
plt.savefig('fusion_model_confidence_distribution.png')
plt.show()

# Find most confused pairs
confusion = {}
for true_label, pred_label in zip(all_labels, all_preds):
    if true_label != pred_label:
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        pair = (true_class, pred_class)
        confusion[pair] = confusion.get(pair, 0) + 1

# Print top confused pairs
print("\nTop confused class pairs (True -> Predicted):")
top_confused = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
for (true_class, pred_class), count in top_confused[:10]:
    print(f"{true_class} -> {pred_class}: {count} times")
