"""
Baseline 5: This baseline is a temporal extension of the
third baseline phase A where individual person features are passed to LSTM
then pooled over all people to recognize group activities.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Person_Activity_Temporal_Classifier(nn.Module):
    """ Phase A: Trains on individual players across 9 frames """

    def __init__(self,input_size = 2048, num_classes=9, hidden_size=256, num_layers=1):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        )

        # Player-Level Temporal Model
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self,x):
        """
        Input x shape: [Batch, Time_Steps, C, H, W]
        """
        batch,frame, c,h,w = x.shape

        # Merge All for resnet50
        x = x.reshape(batch*frame,c,h,w) # (batch * 9, 3, 244, 244)

        # resnet50 Feature Extraction
        features= self.feature_extractor(x) # (batch * 9, 2048, 1, 1)
        features = features.reshape(batch * frame, -1) # (batch * 9, 2048)

        # Group by player for the LSTM
        features_sequence = features.reshape(batch,frame,-1) # (batch,9,2048)

        # LSTM over time for each player
        lstm_out, (hidden,cell) = self.lstm(features_sequence) # (batch,9,hidden_size)

        # Grab the final temporal state
        lstm_out = lstm_out[:, -1, :]  # (batch,Hidden) (take the last frame only)

        # Classification
        out = self.fc(lstm_out) # (batch, 9_Classes)
        return out


class Group_Activity_Classifier(nn.Module):
    def __init__(self,person_feature_extraction,num_classes,hidden_size=256):
        super(Group_Activity_Classifier, self).__init__()

        self.feature_extractor = person_feature_extraction.feature_extractor
        self.lstm = person_feature_extraction.lstm

        for layer in [self.feature_extractor,self.lstm]:
            for param in layer.parameters():
                param.requires_grad = False

        fc_input_dim = (2048 + hidden_size) * 2

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_input_dim, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self,x):
        """
        Input x shape: [Batch, Num_Players, Time_Steps, C, H, W]
        """
        batch,player,frame, c,h,w = x.shape

        # Merge All for resnet50
        x = x.reshape(batch*player*frame,c,h,w) # (batch * 12 * 9, 3, 244, 244)

        # resnet50 Feature Extraction
        features= self.feature_extractor(x) # (batch * 12 * 9, 2048, 1, 1)
        features_sequence = features.reshape(batch * player,frame, -1) # (batch * 12, 9, 2048)

        # LSTM over time for each player
        lstm_out, (hidden,cell) = self.lstm(features_sequence) # (batch*12,9,hidden_size)

        # Grab the final temporal state & central spatial feature for each player and combine them
        final_lstm_out = lstm_out[:, -1, :]  # (batch*12, Hidden) (take the last frame only)
        center_cnn_feature = features_sequence[:, 4, :]  # (batch*12, 2048)

        combined_features = torch.cat([final_lstm_out, center_cnn_feature], dim=1)  # (batch*12, 2048 + Hidden)
        player_combined_features = combined_features.reshape(batch,player,-1) # (batch,12,Hidden+2048)

        max_pooled = torch.max(player_combined_features, dim=1)[0]
        mean_pooled = torch.mean(player_combined_features, dim=1)

        pooled_features = torch.cat([max_pooled, mean_pooled], dim=1) # (batch, 2 * (2048 + Hidden))

        # Group Classification
        out = self.fc(pooled_features)
        return out
