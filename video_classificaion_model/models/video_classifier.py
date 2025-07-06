import torch
import torch.nn as nn
import torchvision.models as models

class VideoClassifier(nn.Module):
    def __init__(self, cnn_type='resnet18', hidden_dim=256, num_layers=1, dropout=0.5):
        super(VideoClassifier, self).__init__()

        # 1. CNN backbone (사전 학습된거 불러와서 쓰기)
        resnet = models.resnet18(pretrained=True)
        # stage4와 fc층 제거
        #modules = [
        #    resnet.conv1,
        #    resnet.bn1,
        #    resnet.relu,
        #    resnet.maxpool,
        #   resnet.layer1,
        #    resnet.layer2,
        #    resnet.layer3,
        #   resnet.avgpool
        #]
        modules = list(resnet.children())[:-1] 
        self.cnn = nn.Sequential(*modules)
        self.feature_dim = 512
        self.feature_reducer = nn.Linear(512, 256)
        self.feature_dim = 256

        # 2. LSTM: input_dim=512 → hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # 3. FC layer → binary classification
        self.fc = nn.Linear(hidden_dim, 1)
        

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] - 비디오 시퀀스
        Returns:
            logits: [B, 1] - sigmoid 통과 전 raw score
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)             # [B*T, C, H, W]

        # CNN → feature vector
        features = self.cnn(x)                 # [B*T, 512, 1, 1]
        features = features.view(B, T, 512)               # [B, T, 512]
        features = self.feature_reducer(features)         # [B, T, 256]
        
        # LSTM → 마지막 timestep hidden
        _, (hn, _) = self.lstm(features)       # hn: [num_layers, B, H]
        last_hidden = hn[-1]                   # [B, H]

        
        out = self.fc(last_hidden)             # [B, 1]
        return out  # BCEWithLogitsLoss()에 바로 넣기 (sigmoid X) 기억하시죠 !!!!
