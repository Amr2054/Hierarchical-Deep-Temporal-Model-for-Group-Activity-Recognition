"""
Baseline 3:
resnet50 model on each player is fine-tuned to recognize
person-level actions. Then, Individual features are pooled across all people
to recognize group activities in a scene
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class PersonLevelClassifier(nn.Module):
    def __init__(self,num_classes=9):
        super(PersonLevelClassifier, self).__init__()
        self.backbone = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.in_features, out_features= num_classes),

        )

    def forward(self, x):
        return self.backbone(x)

class GroupLevelClassifier(nn.Module):
    def __init__(self,person_classifier,num_classes=8):
        super(GroupLevelClassifier, self).__init__()

        self.person_feature_extractor_fc_in = person_classifier.in_features

        layers = list(person_classifier.backbone.children())[:-1]
        self.person_feature_extractor = nn.Sequential(*layers) # remove last FC layer

        for param in self.person_feature_extractor.parameters(): # Feature extractor is frozen
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.person_feature_extractor_fc_in, out_features= 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features= num_classes),
        )

    def forward(self, x):
        batch_size, num_players, C, H, W = x.size() # Batch, player, channels, Height, Width
        x = x.reshape(batch_size * num_players, C, H, W) # (B*P,C,H,W)
        features = self.person_feature_extractor(x) # (B*P,2048,1,1)

        features = features.reshape(batch_size, num_players, -1) # (B,P,2048)
        group_features,_ = torch.max(features,dim=1) # (B,2048)

        out = self.fc(group_features) # (B,num_classes)
        
        return out
