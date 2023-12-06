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
        pass


class WindowModel(SpecModel):
    @classmethod
    def get_model(cls, device, window_size=35, window_stride=10):
        pass


def _Window(model, device, window_size=35, window_stride=10):
    model.lstm = nn.LSTM(1, 1)

    def wrapper(forward):
        def inner(x):
            results = torch.empty(len(x))
            for index, sample in enumerate(x):
                device = sample.device
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

    model.forward = wrapper(model.forward)
    model = model.to(device)
    return model


def _LSTM(model, device):
    model.fc = nn.LSTM(512, 1)

    def wrapper(forward):
        def inner(x):
            return torch.sigmoid(forward(x)[0])

        return inner

    model.forward = wrapper(model.forward)
    model = model.to(device)
    return model


def _Linear(model, device):
    model.fc = nn.Linear(512, 1)

    def wrapper(forward):
        def inner(x):
            return torch.sigmoid(forward(x))

        return inner

    model.forward = wrapper(model.forward)
    model = model.to(device)
    return model


class LSTM(SpecModel):
    @classmethod
    def get_model(cls, device):
        return _LSTM(cls.model, device)


class Linear(SpecModel):
    @classmethod
    def get_model(cls, device):
        return _Linear(cls.model, device)


spec_models: dict[str, SpecModel] = {}
for model_creator, pretrained, window, single_channel in product(
    (LSTM, Linear), ("Pretrained", ""), ("Window", ""), ("SingleChannel", "")
):
    model_name = pretrained + single_channel + window + model_creator.__name__ + "Model"

    if not window and not single_channel:
        continue
    locals()[model_name]: SpecModel = type(
        model_name, (SpecModel,), model_creator.__dict__.copy()
    )

    if window:
        locals()[model_name] = type(
            model_name, (WindowModel,), model_creator.__dict__.copy()
        )

        def _get_model(cls, device, window_size=35, window_stride=10):
            return _Window(cls.model, device, window_size, window_stride)

        locals()[model_name].get_model = classmethod(_get_model)
    locals()[model_name].model = models.resnet18(weights=None)
    if pretrained:
        locals()[model_name].model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if single_channel:

        def wrapper(get_model_function):
            def inner(*args, **kwargs):
                model = get_model_function(*args, **kwargs)
                model.conv1 = nn.Conv2d(
                    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                return model

            return inner

        locals()[model_name].get_model = wrapper(locals()[model_name].get_model)
    spec_models[model_name] = locals()[model_name]
