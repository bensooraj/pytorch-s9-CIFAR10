import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # [INPUT | C1] Convolution Block 1
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.Dropout(dropout_value)
        )

        # [C2] Convolution Block 2
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Dropout(dropout_value)
        )

        # [C3] Convolution Block 3
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.Dropout(dropout_value)
        )

        # [C4] Convolution Block 4
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.Dropout(dropout_value)
        )
        
        # Global Average Pooling
        self.gap = nn.AvgPool2d(kernel_size=32, stride=1, padding=0)

        # Output Block
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=10, bias=True)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        
        x = self.gap(x) # output size: [1, 256, 1, 1]
        x = x.flatten(start_dim=1) # output size: [1, 256]

        x = self.fc1(x)
        # x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    def summary(self, input_size):
        print(summary(self, input_size))
