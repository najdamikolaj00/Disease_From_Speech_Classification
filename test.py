from itertools import product
from pathlib import Path

import torch
from torch import nn

from Evaluation.model_test import test_model
from Evaluation.utilities import check_cuda_availability
from Models.ModelOptions import ModelKernel, TrainingOption, InputChannels, BaseModel, LastLayer
from Models.Models import get_model
from config import data_path

if __name__ == '__main__':
    device = check_cuda_availability()
    disease = 'Rekurrensparese'
    for kernel, training_option, input_channels in product(ModelKernel, TrainingOption, InputChannels):
        model_creator = lambda: get_model(
            BaseModel.ResNet18,
            LastLayer.Linear,
            training_option,
            kernel,
            input_channels,
        )
        model_type = model_creator()
        file_path = (
            data_path
            / f"Lists/Vowels_a{'ll' if 'MultiChannel' in model_type.__name__ else ''}_{disease}_train.txt"
        )
        for model_weighs in Path('Data/results').iterdir():
            if model_type.__name__ not in model_weighs.name:
                continue
            model_type.model.load_state_dict(torch.load(model_weighs))

            test_model(device, file_path, model_type.model, nn.BCELoss())

