import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class ResNet50FineTuner(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet50FineTuner, self).__init__()
        self.backbone = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)