"""
Baseline 6: This model apply feature extraction on fine-tuned
person-level action classifier then pools all people and
apply LSTM to learn group activity
"""
import torch.nn as nn
import torch


class Group_Activity_Temporal_Classifier(nn.Module):
    def __init__(self, person_classifier, num_classes=8,input_size=2048, hidden_size=256, num_layers=1):
        super(Group_Activity_Temporal_Classifier, self).__init__()

        self.person_feature_extractor_fc_in = person_classifier.in_features
        layers = list(person_classifier.backbone.children())[:-1]
        self.person_feature_extractor = nn.Sequential(*layers)  # remove last FC layer

        for param in self.person_feature_extractor.parameters():  # Feature extractor is frozen
            param.requires_grad = False

        self.lstm = nn.LSTM( # input (seq,frames,2048) out (seq,frames,hidden_size)
                            input_size=input_size * 2, # 2048 (Max) + 2048 (Mean) = 4096
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout = 0.5 if num_layers > 1 else 0.0
                            )

        # Dimension Math for the FC Layer:
        # Spatial Anchor: 4096 (Center Frame Max + Mean)
        # Temporal State: (BiLSTM outputs 2 * hidden_size)

        fc_input_dim = (input_size * 2) + (hidden_size * 2)

        self.fc =  nn.Sequential(
            nn.Linear(in_features= fc_input_dim,out_features= 512),
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
        x = x.reshape(batch*player*frame,c,h,w) # (batch * 12 * 9, 3, 244, 244)

        # CNN Feature Extraction
        features= self.person_feature_extractor(x) # (batch * 12 * 9, 2048, 1, 1)
        features = features.reshape(batch, player, frame, -1) # (batch, 12, 9, 2048)

        # Pool across the Players dimension
        max_pool = torch.max(features, dim=1)[0]  # (batch, 9, 2048)
        mean_pool = torch.mean(features, dim=1)  # (batch, 9, 2048)
        pooled_features = torch.cat([max_pool, mean_pool], dim=2)  # (batch, 9, 4096)

        # Temporal Tracking
        lstm_out, (hidden,cell) = self.lstm(pooled_features) # (batch,9,hidden_size)

        # Grab the final temporal state & central spatial features and combine them
        final_lstm_out = lstm_out[:, -1, :]  # (batch, hidden_size*2)
        center_cnn_feature = pooled_features[:, 4, :]  # (batch, 4096)

        # Combine Spatial and Temporal
        combined_features = torch.cat([final_lstm_out, center_cnn_feature], dim=1)  # (batch, 4096 + hidden_size*2)

        # Group Classification
        out = self.fc(combined_features)
        return out

