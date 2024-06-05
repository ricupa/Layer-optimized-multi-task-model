import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import *



class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout1 = nn.Dropout(p=0.6)
        self.fc1 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x[:,0]     #### as it gives the output [bs,1]