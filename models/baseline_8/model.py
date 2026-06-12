import torch
import torch.nn as nn

class Full_Hierarchical_Model_B8(nn.Module):
    def __init__(self, person_classifier, num_classes=8, input_size=2048, hidden1_size=256, hidden2_size=256,
                 num_layers=1):
        super(Full_Hierarchical_Model_B8, self).__init__()

        # STAGE 1: PERSON LEVEL

        # Load Person Feature extractor and lstm_1 (person level)
        self.person_feature_extractor = person_classifier.feature_extractor
        self.lstm_1 = person_classifier.lstm

        # Feature extractor & LSTM 1 is frozen
        for layer in [self.person_feature_extractor, self.lstm_1]:
            for param in layer.parameters():
                param.requires_grad = False

        # STAGE 2: GROUP LEVEL (LSTM 2)

        # Math per player: 2048 (CNN) + 256 (LSTM 1) = 2304
        # Left Team: Max (2304) + Mean (2304) = 4608
        # Right Team: Max (2304) + Mean (2304) = 4608
        # Total Concatenated Court: 9216
        court_feature_size = (input_size + hidden1_size) * 4
        self.team_norm = nn.LayerNorm(court_feature_size)

        self.lstm_2 = nn.LSTM(  # input (seq,frames,2048) out (seq,frames,hidden_size)
            input_size=court_feature_size,
            hidden_size=hidden2_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if num_layers > 1 else 0.0
        )

        # Classification FC Layer
        # Math: 512 (BiLSTM 2 outputs 2 * 256) + 9216 (Center-Frame Spatial Anchor) = 9728
        fc_input_dim = (hidden2_size * 2) + court_feature_size

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_input_dim, out_features=512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        """
        Input x shape: [Batch, Num_Players, Time_Steps, C, H, W]
        """
        batch, player, frame, c, h, w = x.shape

        # Merge All for CNN
        x = x.view(batch * player * frame, c, h, w)  # (batch * 12 * 9, 3, 244, 244)

        # CNN Feature Extraction
        cnn_features = self.person_feature_extractor(x)  # (batch * 12 * 9, 2048, 1, 1)
        cnn_features = cnn_features.view(batch * player, frame, -1)  # (batch * 12 , 9, 2048)

        # LSTM over time for each player
        lstm_out_1, (hidden, cell) = self.lstm_1(cnn_features)  # (batch*12,9,hidden_size)

        # Person Spatial Anchor (Concat CNN + LSTM 1 for EVERY frame)
        combined_person_features = torch.cat([lstm_out_1, cnn_features],
                                             dim=2)  # (batch*12, 9, input_size + hidden_1_size)
        combined_person_features = combined_person_features.view(batch, player, frame, -1)  # (batch,12,9,hidden_size)


        # Left Team (0-5) vs Right Team (6-11)
        left_team = combined_person_features[:, :6, :, :]
        right_team = combined_person_features[:, 6:, :, :]

        # SUB-GROUP POOLING
        left_max = torch.max(left_team, dim=1)[0]
        left_mean = torch.mean(left_team, dim=1)

        right_max = torch.max(right_team, dim=1)[0]
        right_mean = torch.mean(right_team, dim=1)

        # Concatenate Teams: [Left Team features, Right Team features]
        court_sequence = torch.cat([left_max, left_mean, right_max, right_mean], dim=2)  # (batch, 9, 9216)

        # Apply LayerNorm
        court_sequence = self.team_norm(court_sequence)

        # Group Temporal Tracking (LSTM 2)
        lstm_out_2, (hidden, cell) = self.lstm_2(court_sequence)

        # Team Spatial Anchor + Final Classification
        final_temporal = lstm_out_2[:, -1, :]  # (batch, 512)
        center_spatial = court_sequence[:, 4, :]  # (batch, 9216)
        final_features = torch.cat([center_spatial, final_temporal], dim=1)  # (batch, 9216)

        out = self.fc(final_features)
        return out