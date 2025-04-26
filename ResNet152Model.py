
import torch.nn as nn
import torchvision.models as models

# Squeeze-and-Excitation (SE) Block for Attention Mechanism
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class ResNet152Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152Model, self).__init__()
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

        # Save in_features before replacing the fc layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),  # BatchNorm layer present
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # BatchNorm layer present
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)
