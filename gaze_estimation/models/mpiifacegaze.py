from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yacs.config


class Backbone(torchvision.models.ResNet):
    def __init__(self, config: yacs.config.CfgNode):
        block_name = config.model.backbone.resnet_block
        if block_name == 'basic':
            block = torchvision.models.resnet.BasicBlock
        elif block_name == 'bottleneck':
            block = torchvision.models.resnet.Bottleneck

        # configuram cate blocuri resnet de fiecare tip vrem in backbone
        layers = list(config.model.backbone.resnet_layers) + [1] # [2,2,2,1]
        super().__init__(block, layers)

        # stergem ultimele layere de conv, avgpool si fc pentru ca o sa fie inlocuite cu mecanismul de spatial weights
        del self.layer4
        del self.avgpool
        del self.fc

        # selectam ce tip de resnet preantrenat vrem resnet18 de eg.
        pretrained_name = config.model.backbone.pretrained
        if pretrained_name:
            state_dict = torch.hub.load_state_dict_from_url(
                torchvision.models.resnet.model_urls[pretrained_name])
            self.load_state_dict(state_dict, strict=False)
            # While the pretrained models of torchvision are trained
            # using images with RGB channel order, in this repository
            # images are treated as BGR channel order.
            # Therefore, reverse the channel order of the first
            # convolutional layer.
            module = self.conv1
            module.weight.data = module.weight.data[:, [2, 1, 0]]

        ## calculam numarul de canale rezultate ca sa stim exact cate input channels va avea urmatorul layer
        with torch.no_grad():
            data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            features = self.forward(data)
            self.n_features = features.shape[1] # find the number of channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class FaceGaze(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()
        self.feature_extractor = Backbone(config)
        n_channels = self.feature_extractor.n_features

        # spatial weights mechanism
        self.conv = nn.Conv2d(n_channels,
                              1,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        # layer liniar ce va returna unghiurile privirii
        self.fc = nn.Linear(n_channels * 14**2,2) # blocurile CNN vor da un activation volume 14x14xn_channels
        self._register_hook()
        self._initialize_weight()

    def _initialize_weight(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    #  The hook is registered to average the gradients by the number of the channels as in the paper

    # sectiunea 4.1 din paper:The gradient with respect to W
    # is normalised by the total number of the feature maps N,
    # since the weight map affects all the feature maps equally.
    # vezi https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
    def _register_hook(self):
        n_channels = self.feature_extractor.n_features

        def hook(
            module: nn.Module, grad_in: Union[Tuple[torch.Tensor, ...],
                                              torch.Tensor],
            grad_out: Union[Tuple[torch.Tensor, ...], torch.Tensor]
        ) -> Optional[torch.Tensor]:
            return tuple(grad / n_channels for grad in grad_in)

        self.conv.register_backward_hook(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        # spatial weights mechanism se foloseste doar  o convolutie de 1x1 + ReLu ca sa ne dea masca ponderata
        y = F.relu(self.conv(x))
        # masca e inmutita cu fiecare activation map
        x = x * y
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        # returnam pitch, yaw
        return x
