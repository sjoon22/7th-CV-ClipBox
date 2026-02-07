import torch
import torch.nn as nn
from torchvision import models

class VideoClassifier(nn.Module):
    def __init__(self, hidden_dim=256, lstm_layers=1, unfreeze="none"):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # layer4 제거: conv1~layer3까지만 사용
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.feature_dim = 256

        # 기본 freeze
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 원하면 layer3만 unfreeze(부분 학습)
        if unfreeze == "layer3":
            for p in resnet.layer3.parameters():
                p.requires_grad = True

        self.feature_reducer = nn.Linear(self.feature_dim, 256)  # 256 -> 256 (필요 없으면 identity로 바꿔도 됨)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)   # (B*T, 256)

        feats = self.feature_reducer(feats)
        feats = feats.view(B, T, 256)

        out, _ = self.lstm(feats)
        out = out[:, -1, :]
        return self.head(out)
