from typing import Literal
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import tensor
from torchvision.models import ResNet18_Weights


class SpecModel:
    model: nn.Module

    @classmethod
    def get_model(cls, device):
        def wrapper(forward):
            def inner(x):
                return torch.sigmoid(forward(x))

            return inner

        cls.model.forward = wrapper(cls.model.forward)
        model = cls.model.to(device)
        return model


class WindowModel(SpecModel):
    @classmethod
    def get_model(cls, device, window_size=35, window_stride=10):
        def wrapper(forward):
            def inner(x):
                results = torch.empty(len(x))
                device = x.device
                for index, sample in enumerate(x):
                    sample = sample.cpu()
                    for i in range(1, 500):
                        if torch.sum(sample[:, :, -i]) != 0:
                            sample = sample[:, :, :-i]
                            break
                    windows = tensor(
                        np.array(
                            tuple(
                                sample[:, :, i : i + window_size].numpy()
                                for i in range(
                                    0, len(sample[0, -1]) - window_size, window_stride
                                )
                            )
                        )
                    )
                    windows = forward(windows.to(device))
                    results[index] = torch.sigmoid(torch.mean(windows))
                return results.unsqueeze(1).to(device)

            return inner

        cls.model.forward = wrapper(cls.model.forward)
        model = cls.model.to(device)
        return model


def get_module_name(
    base_model: Literal["ResNet18"],
    model_type: Literal["LSTM", "Linear"],
    pretrained: Literal["Pretrained", "NotTrained"],
    window: Literal["Window", "Continuous"],
    single_channel: Literal["SingleChannel", "MultiChannel"],
):
    return pretrained + single_channel + window + model_type + base_model + "BasedModel"


spec_models: dict[str, SpecModel] = {}
for base_model, model_type, pretrained, window, single_channel in product(
    ("ResNet18",),
    ("LSTM", "Linear"),
    ("Pretrained", "NotTrained"),
    ("Window", "Continuous"),
    ("SingleChannel", "MultiChannel"),
):
    model_name = get_module_name(
        base_model, model_type, pretrained, window, single_channel
    )

    locals()[model_name]: SpecModel = type(
        model_name, (WindowModel if window == "Window" else SpecModel,), {}
    )
    if base_model == "ResNet18":
        locals()[model_name].model = models.resnet18(weights=None)
        if pretrained:
            locals()[model_name].model = models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
    if model_type == "Linear":
        locals()[model_name].model.fc = nn.Linear(512, 1)
    elif model_type == "LSTM":
        locals()[model_name].model.fc = nn.LSTM(512, 1)
    if single_channel == "SingleChannel":
        locals()[model_name].conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

    spec_models[model_name] = locals()[model_name]
