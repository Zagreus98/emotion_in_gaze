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
            self.emotion_classifier = nn.Sequential(self.model.classifier)
            self.num_features = self.model.num_features
            self.gaze_regressor = nn.Sequential(nn.Linear(in_features=self.num_features, out_features=2))

        elif config.train.gaze_pretrained:
            self.model = timm.create_model(
                config.model.backbone.name,
                pretrained=True,
                num_classes=2,
            )
            state_dict = torch.load(r'D:\models\finetuned_eth-xgaze_resnet18.pth', map_location='cpu')
            self.model.load_state_dict(state_dict['model'])
            model_path = r'D:\models\finetuned_eth-xgaze_resnet18.pth'
            logger.info(f'Loading pretrained emotion model {model_path}')

            self.gaze_regressor = nn.Sequential(self.model.fc)
            self.num_features = self.model.num_features
            self.emotion_classifier = nn.Sequential(nn.Linear(in_features=self.num_features, out_features=7))
        else:
            self.model = timm.create_model(
                config.model.backbone.name,
                pretrained=True,
                num_classes=7,
            )
            self.emotion_classifier = nn.Sequential(self.model.classifier)
            self.num_features = self.model.num_features
            self.gaze_regressor = nn.Sequential(nn.Linear(in_features=self.num_features, out_features=2))


    def forward(self, x):
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        gaze = self.gaze_regressor(features)
        emotion = self.emotion_classifier(features)
        return gaze, emotion
