import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction = 16):  
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        reduction = max(4,channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return F.relu(out)

class ChannelBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(1, 32),
            ResBlock(32, 64, downsample=True),
            ResBlock(64, 128, downsample=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):  
        out = self.net(x)  
        return out.view(out.size(0), -1)  # [B, 128]


class CNNWaveletDualClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.branch_s2 = ChannelBranch()
        self.branch_s3 = ChannelBranch()
        self.feature_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()

        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 + 64, 128),  
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, s2, s3, feat):
        if feat.dim() == 1:
            feat = feat.unsqueeze(1)   # [B,1]
    
        f = self.feature_fc(feat)      # [B,64]
    
        x_s2 = self.branch_s2(s2)      # [B,128]
        x_s3 = self.branch_s3(s3)      # [B,128]
    
        x = torch.cat([x_s2, x_s3, f], dim=1)   # [B,320]
        return self.classifier(x)
