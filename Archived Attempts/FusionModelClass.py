import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Define the fusion model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        # Load pre-trained models
        self.efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

        # Remove classifiers
        self.efficientnet_b0.classifier = nn.Identity()
        self.resnet152.fc = nn.Identity()

        # Adaptive pooling to get consistent feature size
        self.eff_pool = nn.AdaptiveAvgPool2d(1)
        self.res_pool = nn.AdaptiveAvgPool2d(1)

        # Get feature dimensions
        eff_features = self.efficientnet_b0.features[-1][0].out_channels  # Last Conv layer output
        res_features = self.resnet152.layer4[-1].conv3.out_channels  # Last Conv layer output

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(eff_features + res_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        eff_features = self.efficientnet_b0.features(x)  # Get feature maps
        eff_features = self.eff_pool(eff_features)  # Pool to 1x1
        eff_features = torch.flatten(eff_features, start_dim=1)  # Flatten

        res_features = self.resnet152(x)  # Extract ResNet features
        res_features = self.res_pool(res_features)  # Pool to 1x1
        res_features = torch.flatten(res_features, start_dim=1)  # Flatten

        combined = torch.cat((eff_features, res_features), dim=1)
        return self.classifier(combined)

# Hybrid loss function
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.75, weight=None, label_smoothing=0.1):
        super(HybridLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.alpha = alpha

    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)

        # Calculate F1-inspired component
        probs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()

        # Precision-like term
        precision = (probs * targets_one_hot).sum(dim=0) / (probs.sum(dim=0) + 1e-7)

        # Recall-like term
        recall = (probs * targets_one_hot).sum(dim=0) / (targets_one_hot.sum(dim=0) + 1e-7)

        # F1-like term
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1_loss = 1 - f1.mean()

        # Combined loss
        return self.alpha * ce_loss + (1 - self.alpha) * f1_loss
