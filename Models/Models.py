import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import tensor
from torchvision.models import ResNet18_Weights

from Models.ModelOptions import BaseModel, LastLayer, ModelKernel, InputChannels, TrainingOption
from Models.SpecNetModels import SpecNet, SpecNetWithSE


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
                                sample[:, :, i: i + window_size].numpy()
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


def _get_model_name(
    base_model: BaseModel,
    last_layer_type: LastLayer,
    pretrained: TrainingOption,
    kernel: ModelKernel,
    input_channel: InputChannels,
):
    return ((pretrained if base_model == BaseModel.ResNet18 else TrainingOption.TrainedFromScratch).value
            + input_channel.value + kernel.value + last_layer_type.value + base_model.value + "BasedModel")


def get_model_type(base_model: BaseModel,
                   last_layer_type: LastLayer,
                   pretrained: TrainingOption,
                   kernel: ModelKernel,
                   input_channel: InputChannels,
                   ) -> SpecModel:
    model_name = _get_model_name(base_model, last_layer_type, pretrained, kernel, input_channel)
    model_type: SpecModel = type(
        model_name, (WindowModel if kernel == ModelKernel.Window else SpecModel,), {}
    )
    if base_model == BaseModel.ResNet18:
        model_type.model = models.resnet18(weights=None)
        if pretrained:
            model_type.model = models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
    elif base_model == BaseModel.SpecNet:
        model_type.model = SpecNet()
    elif base_model == BaseModel.SpecNetWithSE:
        model_type.model = SpecNetWithSE()
    model_output_size = 512 if base_model == BaseModel.ResNet18 else 59040
    if last_layer_type == LastLayer.Linear:
        model_type.model.fc = nn.Linear(model_output_size, 1)
    elif last_layer_type == LastLayer.LSTM:
        model_type.model.fc = nn.LSTM(model_output_size, 1)
    if input_channel == InputChannels.SingleChannel and base_model == BaseModel.ResNet18:
        model_type.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
    elif input_channel == InputChannels.MultiChannel and base_model != BaseModel.ResNet18:
        model_type.model.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0
        )
    return model_type
