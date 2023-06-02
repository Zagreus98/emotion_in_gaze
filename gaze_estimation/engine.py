import torch

from .models import GazeEmotion
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    model = GazeEmotion(config)
    # if config.train.emo_pretrained:
    #     set_parameter_requires_grad(model, requires_grad=False)
    #     set_parameter_requires_grad(model.gaze_regressor, requires_grad=True)
    device = torch.device(config.device)
    model.to(device)

    return model


def set_parameter_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad
