import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class Group_Activity_Temporal_Classifier(nn.Module):
    def __init__(self,num_classes,input_size=2048, hidden_size=256, num_layers=3):
        super(Group_Activity_Temporal_Classifier, self).__init__()

        image_feature_extractor = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        # for param in image_feature_extractor.parameters():
        #     param.requires_grad = False
        #
        # for param in image_feature_extractor.layer4.parameters():
        #     param.requires_grad = True

        layers = list(image_feature_extractor.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)  # remove last FC layer


        self.lstm = nn.LSTM( # input (seq,frames,2048) out (seq,frames,hidden_size)
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5 if num_layers > 1 else 0.0
                            )

        fc_input_dim = input_size + (hidden_size * 2)
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

        # Input Shape : (batch(seq), 9, 3, 244, 244)
        seq, frame, c,h,w = x.shape

        x_spatial = x.view(seq *frame,c,h,w) # (seq * 9, 3, 244, 244)
        x_spatial = self.feature_extractor(x_spatial) # (seq * 9, 2048, 1, 1)
        x_spatial = x_spatial.view(seq,frame,-1) # (seq,9,2048)

        x_temporal, (hidden,cell) = self.lstm(x_spatial) # (seq,9,hidden_size)

        # Final temporal state
        final_temporal = x_temporal[:, -1, :]  # (batch, 512 (hidden*2 -> bidirectional LSTM))

        # Center Frame Spatial Anchor (Index 4)
        center_spatial = x_spatial[:, 4, :]  # (batch, 2048)

        # combine both temporal and spatial
        x_total = torch.cat([center_spatial, final_temporal], dim=1)  # (batch, 2560)

        out = self.fc(x_total)

        return out


