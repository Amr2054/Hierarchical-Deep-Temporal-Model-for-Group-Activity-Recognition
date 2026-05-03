import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class Player_Activity_Temporal_Classifier(nn.Module):
    def __init__(self,num_classes,input_size=2048, hidden_size=256, num_layers=3):
        super(Player_Activity_Temporal_Classifier, self).__init__()

        resnet = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        layers = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)  # remove last FC layer

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM( # input (seq,frames,2048) out (seq,frames,hidden_size)
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout = 0.5 if num_layers > 1 else 0.0
                            )

        self.fc =  nn.Sequential(
            nn.Linear(in_features= input_size+hidden_size,out_features= 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 512,out_features= 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 128, out_features= num_classes),
        )

    def forward(self,x):
        """
        Input x shape: [Batch, Num_Players, Time_Steps, C, H, W]
        """
        batch,player,frame, c,h,w = x.shape

        # Merge All for CNN
        x = x.view(batch*player*frame,c,h,w) # (seq * 12 * 9, 3, 244, 244)

        # CNN Feature Extraction
        features= self.feature_extractor(x) # (seq * 12 * 9, 2048, 1, 1)
        features = features.view(batch * player * frame, -1) # (seq * 12 * 9, 2048)

        # Group by player for the LSTM
        features_sequence = features.view(batch*player,frame,-1) # (batch*12,9,2048)

        # LSTM over time for each player
        lstm_out, (hidden,cell) = self.lstm(features_sequence) # (batch*12,9,hidden_size)

        # Grab the final temporal state & last spatial feature for each player and combine them
        final_lstm_out = lstm_out[:, -1, :]  # (batch*12, Hidden) (take the last frame only)
        last_cnn_feature = features_sequence [:, -1, :] # (batch*12, 2048)
        combined_features = torch.cat([final_lstm_out,last_cnn_feature],dim=1) # (batch*12,hidden_size+2048)

        # Max Pooling in players dimension
        player_combined_features = combined_features.view(batch,player,-1) # (batch,player (12), hidden_size+2048)
        pooled_features = torch.max(player_combined_features,dim=1) # (batch, hidden_size+2048)

        # Group Classification
        out = self.fc(pooled_features)
        return out
