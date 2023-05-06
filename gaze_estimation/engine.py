import torch

from .models import FaceGaze, GazeEmotion
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    mode = config.mode

    if mode == 'MPIIFaceGaze':
        model = FaceGaze(config)
    elif mode == 'ETHXGaze':
        model = GazeEmotion(config)
        if config.train.resume_path:
            state_dict = torch.load(config.train.resume_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])
        # if config.train.emo_pretrained:
        #     set_parameter_requires_grad(model, requires_grad=False)
        #     set_parameter_requires_grad(model.gaze_regressor, requires_grad=True)

    else:
        raise ValueError
    device = torch.device(config.device)
    model.to(device)

    return model


def set_parameter_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad
