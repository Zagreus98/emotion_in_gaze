import torch.nn as nn
import yacs.config
import torch

from .types import LossType


class TotalLoss(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()
        class_weights = torch.Tensor(config.train.class_weights).to(torch.device(config.device))
        self.config = config
        self.emotion_loss = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=-1,
            label_smoothing=config.train.label_smoothing,
            weight=class_weights,
        )
        self.gaze_loss = self.create_gaze_loss()

    def create_gaze_loss(self) -> nn.Module:
        loss_name = self.config.train.loss
        if loss_name == LossType.L1.name:
            return nn.L1Loss(reduction='none')
        elif loss_name == LossType.L2.name:
            return nn.MSELoss(reduction='none')
        elif loss_name == LossType.SmoothL1.name:
            return nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError

    def forward(self, pred_gazes, pred_emotions, grd_gazes, grd_emotions):
        weights = self.config.train.task_weights
        # compute emotion classification loss
        e_loss = self.emotion_loss(pred_emotions, grd_emotions)
        if torch.isnan(e_loss):
            e_loss = torch.tensor(0.0)

        # compute gaze loss
        g_loss = self.gaze_loss(pred_gazes, grd_gazes[:, 1:])  # compute metric
        g_loss *= grd_gazes[:, 0].unsqueeze(1)  # multiply with ignore flags
        nr_of_non_zero_elem = torch.sum(grd_gazes[:, 0]) * 2
        if nr_of_non_zero_elem > 0:
            g_loss = torch.sum(g_loss) / nr_of_non_zero_elem
        else:
            g_loss = torch.tensor(0.0)

        total_loss = weights['emotion'] * e_loss + weights['gaze'] * g_loss

        return total_loss, e_loss, g_loss
