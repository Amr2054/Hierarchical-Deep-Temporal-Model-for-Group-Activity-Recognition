import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50


class Group_Activity_Temporal_Classifier(nn.Module):
    def __init__(self,num_classes,input_size=2048, hidden_size=256, num_layers=3):
        super(Group_Activity_Temporal_Classifier, self).__init__()

        image_feature_extractor = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        layers = list(image_feature_extractor.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)  # remove last FC layer

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM( # input (seq,frames,2048) out (seq,frames,hidden_size)
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout = 0.5 #TODO Investigate
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

        # Input Shape : (batch(seq), 9, 3, 244, 244)
        seq, frame, c,h,w = x.shape

        x_spatial = x.view(seq *frame,c,h,w) # (seq * 9, 3, 244, 244)
        x_spatial = self.feature_extractor(x_spatial) # (seq * 9, 2048, 1, 1)
        x_spatial = x_spatial.view(seq,frame,-1) # (seq,9,2048)

        x_temporal, (hidden,cell) = self.lstm(x_spatial) # (seq,9,hidden_size)

        x_total = torch.cat([x_spatial,x_temporal],dim=2) # (seq,9,hidden_size+2048)
        x_total = x_total[:,-1, :] #(seq, hidden_size+2048) (take the last frame only)
        x_total = self.fc(x_total)

        return x_total


