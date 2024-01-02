from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import tensor
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from Models.ModelOptions import (
    BaseModel,
    LastLayer,
    ModelKernel,
    InputChannels,
    TrainingOption,
)
from Models.SpecNetModels import SpecNet, SpecNetWithSE


def update_forward(model_function):
    def wrapper(*args, **kwargs):
        def forward_wrapper(forward_function):
            def inner(x):
                return torch.sigmoid(forward_function(x))

            return inner

        model = model_function(*args, **kwargs)
        model.forward = forward_wrapper(model.forward)
        return model

    return wrapper


def set_last_layer(model_function, last_layer: LastLayer, model_output_size: int):
    def wrapper(*args, **kwargs):

        model = model_function(*args, **kwargs)
        if last_layer == LastLayer.Linear:
            model.fc = nn.Linear(model_output_size, 1)
        elif last_layer == LastLayer.LSTM:
            model.fc = nn.LSTM(model_output_size, 1)
        return model

    return wrapper


def set_single_channel(model_function):
    def wrapper(*args, **kwargs):
        model = model_function(*args, **kwargs)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        return model

    return wrapper


def set_multi_channel(model_function):
    def wrapper(*args, **kwargs):
        model = model_function(*args, **kwargs)
        model.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0
        )
        return model

    return wrapper


def set_kernel_to_window(model_function: Callable, window_size: int, window_stride: int):
    def wrapper(*args, **kwargs):
        def forward_wrapper(forward):
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
                                sample[:, :, i: i + window_size].numpy()
                                for i in range(
                                    0, len(sample[0, -1]) - window_size, window_stride
                                )
                            )
                        )
                    )
                    windows = forward(
                        transforms.Resize((224, 224), antialias=None)(windows).to(
                            device
                        )
                    )
                    results[index] = torch.sigmoid(torch.mean(windows))
                return results.unsqueeze(1).to(device)

            return inner

        model = model_function(*args, **kwargs)
        model.forward = forward_wrapper(model.forward)
        return model

    return wrapper


def get_model_name(
    base_model: BaseModel,
    last_layer_type: LastLayer,
    pretrained: TrainingOption,
    kernel: ModelKernel,
    input_channel: InputChannels,
):
    return (
        (
            pretrained
            if base_model == BaseModel.ResNet18
            else TrainingOption.TrainedFromScratch
        ).value
        + input_channel.value
        + kernel.value
        + last_layer_type.value
        + base_model.value
        + "BasedModel"
    )


def get_model(
    base_model: BaseModel,
    last_layer_type: LastLayer,
    pretrained: TrainingOption,
    kernel: ModelKernel,
    input_channel: InputChannels,
    window_size: int = None,
    window_stride: int = None,
    *_
) -> nn.Module:
    model_name = get_model_name(
        base_model, last_layer_type, pretrained, kernel, input_channel
    )

    kwargs = {}
    if base_model == BaseModel.ResNet18:
        kwargs = {"weights": None}
        model_function = models.resnet18
        if pretrained:
            kwargs = {"weights": ResNet18_Weights.DEFAULT}
    elif base_model == BaseModel.SpecNet:
        model_function = SpecNet
    elif base_model == BaseModel.SpecNetWithSE:
        model_function = SpecNetWithSE
    else:
        raise ValueError
    model_output_size = 512 if base_model == BaseModel.ResNet18 else 46656
    model_function = set_last_layer(model_function, last_layer_type, model_output_size)
    if (
        input_channel == InputChannels.SingleChannel
        and base_model == BaseModel.ResNet18
    ):
        model_function = set_single_channel(model_function)
    elif (
        input_channel == InputChannels.MultiChannel and base_model != BaseModel.ResNet18
    ):
        model_function = set_multi_channel(model_function)
    model_function = update_forward(model_function)
    if kernel == ModelKernel.Window:
        model_function = set_kernel_to_window(model_function, window_size, window_stride)
    model = model_function(**kwargs)
    model.__name__ = model_name
    return model
