import torch
import torch.nn as nn
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

class EfficientNetB5Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB5Model, self).__init__()
        self.efficientnet = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.efficientnet.classifier[1].in_features, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


# 1) Instantiate and load weights
model = EfficientNetB5Model(num_classes=18)
model.load_state_dict(torch.load("best_efficientnet_b5_model.pth"))
model.eval()

# 2) Export to ONNX
dummy = torch.randn(1, 3, 460, 460)
torch.onnx.export(
    model, dummy, "efficientnet_b5.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
