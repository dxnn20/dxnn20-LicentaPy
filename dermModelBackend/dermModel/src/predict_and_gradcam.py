import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
import sys

# Configuration
CONFIG = {
    "image_size": (460, 460),
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

# Custom Grad-CAM for EfficientNet-B5
class EfficientNetB5GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()
        return cam, target_class

# EfficientNet-B5 model wrapper
class EfficientNetB5Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB5Model, self).__init__()
        self.efficientnet = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.efficientnet.classifier[1].in_features, 512),
            nn.SiLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

# Preprocess
def load_and_preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(CONFIG["device"])
    return img, tensor

# Run prediction and Grad-CAM manually
def predict_and_gradcam(model, image_path, target_layer):
    img, input_tensor = load_and_preprocess(image_path)

    gradcam = EfficientNetB5GradCAM(model, target_layer)
    cam, pred = gradcam.generate(input_tensor)

    rgb_img = np.array(img).astype(np.float32) / 255.0
    heatmap = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
    cmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET )
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    overlay = (0.4 * cmap[..., ::-1] + 0.6 * (rgb_img * 255)).astype(np.uint8)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)
        top3 = torch.topk(probs, k=3).indices.squeeze().tolist()
        confidence = probs[0, pred].item()

    return {
        "predicted_class": pred,
        "confidence": confidence,
        "top3": top3,
        "overlay_image": overlay
    }

# Main CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="E:\\dxnn20-LicentaPy\\dermModelBackend\\dermModel\\src\\best_efficientnet_b5_model.pth")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model = EfficientNetB5Model(18).to(CONFIG["device"])
    model.load_state_dict(torch.load(args.model, map_location=CONFIG["device"]))

    target_layer = model.efficientnet.features[-1]
    result = predict_and_gradcam(model, args.input, target_layer)

    cv2.imwrite(args.output, result["overlay_image"])

    print(json.dumps({
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "top3": result["top3"]
    }))
