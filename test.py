import sys
from itertools import product
from pathlib import Path

import torch

from Evaluation.model_test import test_model
from Evaluation.utilities import check_cuda_availability
from Models.ModelOptions import ModelKernel, TrainingOption, InputChannels, BaseModel, LastLayer
from Models.Models import get_model
from config import data_path

if __name__ == '__main__':
    sys.stdout = open('results.txt', 'w')
    device = check_cuda_availability()
    disease = 'Rekurrensparese'
    for kernel, training_option, input_channels in product(ModelKernel, TrainingOption, InputChannels):
        model = get_model(
            BaseModel.ResNet18,
            LastLayer.Linear,
            training_option,
            kernel,
            input_channels,
        )
        file_path = (
            data_path
            / f"Lists/Vowels_a{'ll' if input_channels == InputChannels.MultiChannel else ''}_{disease}_test.txt"
        )
        for model_weighs in Path('Data/results').iterdir():
            if model.__name__ not in model_weighs.name:
                continue
            print(model.__name__)
            model.load_state_dict(torch.load(model_weighs, map_location=device))
            test_model(device, file_path, model)
            sys.stdout.flush()
    sys.stdout.close()

