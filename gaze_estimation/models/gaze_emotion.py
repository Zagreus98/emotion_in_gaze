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
        self.dropout = nn.Dropout(p=0.1)
        if config.train.emo_pretrained:
            self.model = torch.load(r'D:\models\enet_b0_7.pt', map_location='cpu')
            model_path = r'D:\models\enet_b0_7.pt'
            logger.info(f'Loading pretrained emotion model {model_path}')
            self.emotion_classifier = nn.Sequential(self.model.classifier)
            self.num_features = self.model.num_features
            self.gaze_regressor = nn.Sequential(nn.Linear(in_features=self.num_features, out_features=2))
            # # spatial weights mechanism
            # self.conv = nn.Conv2d(self.num_features,
            #                       1,
            #                       kernel_size=1,
            #                       stride=1,
            #                       padding=0)
            # self._register_hook()

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
            # # spatial weights mechanism
            # self.conv = nn.Conv2d(self.num_features,
            #                       1,
            #                       kernel_size=1,
            #                       stride=1,
            #                       padding=0)
            self.emotion_classifier = nn.Sequential(nn.Linear(in_features=self.num_features, out_features=7))
            # self._register_hook()
        else:
            self.model = timm.create_model(
                config.model.backbone.name,
                pretrained=True,
                num_classes=7,
            )
            # self.model = torch.load(r'D:\models\enet_b0_7.pt', map_location='cpu')
            self.emotion_classifier = nn.Sequential(self.model.classifier)
            self.num_features = self.model.num_features
            self.gaze_regressor = nn.Sequential(nn.Linear(in_features=self.num_features, out_features=2))
            # self.conv = nn.Conv2d(self.num_features,
            #                       1,
            #                       kernel_size=1,
            #                       stride=1,
            #                       padding=0)
            # self._register_hook()

    def _register_hook(self):
        n_channels = self.num_features

        def hook(module, grad_in, grad_out):
            return tuple(grad / n_channels for grad in grad_in)

        self.conv.register_backward_hook(hook)

    def forward(self, x):
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        features = self.dropout(features)
        # attention_map = self.conv(features)
        # features = features * attention_map
        # features = self.model.global_pool(features)
        gaze = self.gaze_regressor(features)
        emotion = self.emotion_classifier(features)
        return gaze, emotion
