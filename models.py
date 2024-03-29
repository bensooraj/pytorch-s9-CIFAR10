import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self, debug=False):
        super(Net, self).__init__()
        self.debug = debug
        # [INPUT | C1] Convolution Block 1
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        # Depthwise Separable Convolution
        # [C2] Convolution Block 2 
        self.convBlock2 = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, groups=32, bias=True),
            # Pointwise Convolution
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        # [C3] Convolution Block 3
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        # [C4] Convolution Block 4
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Global Average Pooling
        self.gap = nn.AvgPool2d(kernel_size=15, stride=1, padding=0)

        # Output Block
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10, bias=True)
        )

    def forward(self, x):
        if self.debug:
            print("[INPUT]", x.shape)
        
        x = self.convBlock1(x)
        if self.debug:
            print("[AFTER C1]", x.shape)
        
        x = self.convBlock2(x)
        if self.debug:
            print("[AFTER C2]", x.shape)
        
        x = self.convBlock3(x)
        if self.debug:
            print("[AFTER C3]", x.shape)

        x = self.convBlock4(x)
        if self.debug:
            print("[AFTER C4]", x.shape)

        x = self.gap(x) # output size: [1, 128, 1, 1]
        if self.debug:
            print("[AFTER GAP]", x.shape)

        x = x.flatten(start_dim=1) # output size: [1, 128]
        if self.debug:
            print("[AFTER Flatten]", x.shape)

        x = self.fc1(x)
        if self.debug:
            print("[AFTER FC1]", x.shape)

        return F.log_softmax(x, dim=-1)

    def summary(self, input_size):
        print(summary(self, input_size))
