import numpy as np
import torchvision.models as models
from torch import tensor
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn


def _Window(model, device):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, 1)
    window_size = 40
    window_stride = 10

    def wrapper(forward):
        def inner(x):
            forwarded_samples = []
            for sample in x:
                for i in range(1, 500):
                    if torch.sum(sample[:, :, -i]) != 0:
                        sample = sample[:, :, :-i]
                        break
                windows = tensor(tuple(sample[:, :, i:i + window_size].numpy() for i in range(
                    0, len(sample[0, -1]) - window_size, window_stride)))
                forward(windows)
            return tensor(forwarded_samples).unsqueeze(1)

        return inner

    model.forward = wrapper(model.forward)
    model = model.to(device)
    return model


def _LSTM(model, device):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
    model.fc = nn.LSTM(512, 1)

    def wrapper(forward):
        def inner(x):
            return torch.sigmoid(forward(x)[0])

        return inner

    model.forward = wrapper(model.forward)
    model = model.to(device)
    return model


def _Linear(model, device):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, 1)

    def wrapper(forward):
        def inner(x):
            return torch.sigmoid(forward(x))

        return inner

    model.forward = wrapper(model.forward)
    model = model.to(device)
    return model


def PreLSTM(device):
    return _LSTM(models.resnet18(weights=ResNet18_Weights.DEFAULT), device)


def LSTM(device):
    return _LSTM(models.resnet18(), device)


def PreLinear(device):
    return _Linear(models.resnet18(weights=ResNet18_Weights.DEFAULT), device)


def Linear(device):
    return _Linear(models.resnet18(), device)


def PreWindow(device):
    return _Window(models.resnet18(weights=ResNet18_Weights.DEFAULT), device)


def Window(device):
    return _Window(models.resnet18(), device)
