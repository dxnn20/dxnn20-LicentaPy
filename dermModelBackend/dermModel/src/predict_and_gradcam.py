import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CONFIG = {
    "image_size": (470, 470),
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

class EfficientNetB5Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB5Model, self).__init__()
        self.efficientnet = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.efficientnet.classifier[1].in_features, 512),
            nn.GELU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

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

def predict_and_gradcam(model, image_path, target_layer):
    img, input_tensor = load_and_preprocess(image_path)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        top3 = torch.topk(probs, k=3).indices.squeeze().tolist()

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    grayscale_cam = cv2.resize(grayscale_cam, (img.width, img.height))

    rgb_img = np.array(img).astype(np.float32) / 255.0
    overlay_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top3": top3,
        "overlay_image": overlay_image[..., ::-1]  # RGB to BGR for cv2.imwrite
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="D:\\LICENTAPY\\dermModelBackend\\dermModel\\src\\best_efficientnet_b5_model.pth")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model = EfficientNetB5Model(10).to(CONFIG["device"])
    model.load_state_dict(torch.load(args.model, map_location=CONFIG["device"]))
    model.eval()

    target_layer = model.efficientnet.features[-1]
    result = predict_and_gradcam(model, args.input, target_layer)

    cv2.imwrite(args.output, result["overlay_image"])

    print(json.dumps({
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "top3": result["top3"]
    }))
