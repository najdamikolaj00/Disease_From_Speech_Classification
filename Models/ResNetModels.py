import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import tensor
from torchvision.models import ResNet18_Weights


def _Window(model, device, window_size=35, window_stride=10):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, 1)
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
                windows = tensor(np.array(tuple(sample[:, :, i:i + window_size].numpy() for i in range(
                    0, len(sample[0, -1]) - window_size, window_stride))))
                windows = forward(windows.to(device))
                results[index] = torch.sigmoid(torch.mean(windows))
            return results.unsqueeze(1).to(device)

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


def _Linear_single_channel(model, device):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
    return _Linear(model, device)


def _Linear(model, device):
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
    return _Linear_single_channel(models.resnet18(weights=ResNet18_Weights.DEFAULT), device)


def Linear(device):
    return _Linear_single_channel(models.resnet18(), device)


def PreLinearMultichannel(device):
    return _Linear(models.resnet18(weights=ResNet18_Weights.DEFAULT), device)


def LinearMultichannel(device):
    return _Linear_single_channel(models.resnet18(), device)


def PreWindow(device, window_size=35, window_stride=10):
    return _Window(models.resnet18(weights=ResNet18_Weights.DEFAULT), device, window_size, window_stride)


def Window(device, window_size=35, window_stride=10):
    return _Window(models.resnet18(), device, window_size, window_stride)
