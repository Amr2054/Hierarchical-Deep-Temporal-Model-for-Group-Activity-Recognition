import torch.nn as nn
import torchvision
from torchvision.models import resnet50
import torch

class PersonLevelClassifier(nn.Module):
    def __init__(self,num_classes=9):
        super(PersonLevelClassifier, self).__init__()
        self.backbone = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

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
        print(f"Person feature extractor output dimension: {self.person_feature_extractor_fc_in}")

        layers = list(person_classifier.backbone.children())[:-1]
        self.person_feature_extractor = nn.Sequential(*layers) # remove last FC layer

        for param in self.person_feature_extractor.parameters(): # Feature extractor is frozen
            param.requires_grad = False

        # self.pool = nn.AdaptiveAvgPool2d((1,2048)) # from (12,2048) (players,features) -> (1,2048) (Feature vector for the group)


        self.fc = nn.Sequential(
            nn.Linear(in_features=self.person_feature_extractor_fc_in, out_features= 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features= num_classes),
        )

    def forward(self, x):
        batch_size, num_players, C, H, W = x.size()
        x = x.view(batch_size * num_players, C, H, W) # (B*P,C,H,W)
        features = self.person_feature_extractor(x) # (B*P,2048,1,1)

        features = features.view(batch_size, num_players, -1) # (B,P,2048)
        # group_features = self.pool(features.permute(0,2,1)) # (B,2048,1)

        group_features,_ = torch.max(features,dim=1) # (B,2048)

        out = self.fc(group_features)
        
        return out
