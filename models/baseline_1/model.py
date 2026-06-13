"""
Baseline B1 : This baseline is the basic resnet50
model fine-tuned for group activity recognition in a single frame.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class ResNet50FineTuner(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet50FineTuner, self).__init__()
        self.backbone = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        fc_input_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        return self.backbone(x)