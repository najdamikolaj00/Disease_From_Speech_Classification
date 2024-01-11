from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import tensor
from torchvision import transforms


def _window_forward_wrapper(forward, window_size: int, window_stride: int):
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
            windows = forward(
                transforms.Resize((224, 224), antialias=None)(windows).to(device)
            )
            results[index] = torch.sigmoid(torch.mean(windows))
        return results.unsqueeze(1).to(device)

    return inner


def _forward_wrapper(forward_function):
    def inner(x):
        return torch.sigmoid(forward_function(x))

    return inner


def adjust(
    model_creation_function: Callable[[], nn.Module],
    multichannel: bool,
    window: bool,
    window_size: int = None,
    window_stride: int = None,
) -> Callable[[], nn.Module]:
    def wrapper():
        model = model_creation_function()
        model.__name__ = "SingleChannel"
        if not multichannel:
            model.conv1 = nn.Conv2d(
                1,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            model.__name__ = "MultiChannel"
        model.fc = nn.Linear(512, 1)

        model.forward = (
            partial(
                _window_forward_wrapper,
                window_size=window_size,
                window_stride=window_stride,
            )
            if window
            else _forward_wrapper
        )(model.forward)
        model.__name__ += "Window" if window else "Traditional"
        model.__name__ += type(model).__name__
        return model

    return wrapper
