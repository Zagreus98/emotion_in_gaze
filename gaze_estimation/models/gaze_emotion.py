from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
import yacs.config

logger = logging.getLogger(__name__)


class GazeEmotion(torch.nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()
        if config.train.emo_pretrained:
            self.model = torch.load(r'D:\models\enet_b2_7.pt', map_location='cpu')
            model_path = r'D:\models\enet_b2_7.pt'
            logger.info(f'Loading pretrained emotion model {model_path}')
        else:
            self.model = timm.create_model(
                config.model.backbone.name,
                pretrained=True,
                num_classes=7,
            )
        self.emotion_classifier = nn.Sequential(self.model.classifier)
        self.gaze_regressor = nn.Sequential(nn.Linear(in_features=1408, out_features=2))

    def forward(self, x):
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        gaze = self.gaze_regressor(features)
        emotion = self.emotion_classifier(features)
        return gaze, emotion
