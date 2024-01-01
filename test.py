from itertools import product

from torch import nn
from Models.ModelOptions import (
    BaseModel,
    LastLayer,
    ModelKernel,
    InputChannels,
    TrainingOption,
)
from Evaluation.training_validation import training_validation
from Evaluation.utilities import check_cuda_availability
from Models.Models import get_model_type, WindowModel
from config import data_path, writer

if __name__ == "__main__":
    device = check_cuda_availability()
    disease = "Rekurrensparese"

    criterion = nn.BCELoss()

    for kernel, training_option, input_channels in product(ModelKernel, TrainingOption, InputChannels):
        model_creator = lambda: get_model_type(
            BaseModel.ResNet18,
            LastLayer.Linear,
            training_option,
            kernel,
            input_channels,
        )
        model_type = model_creator()

        augmentation_types = [
            ("augmentation", "frequency_masking"),
            ("augmentation", "time_masking"),
            ("augmentation", "combined_masking"),
            ("augmentation", "no_augmentation"),
        ]

        hyperparameter_combinations = product(augmentation_types, batch_size_candidates)

        if isinstance(model_type, WindowModel):
            hyperparameter_combinations = product(
                batch_size_candidates,
                window_length_candidates,
                window_stride_candidates,
            )
        else:
            hyperparameter_combinations = product(
                augmentation_types,
                batch_size_candidates,
            )

        for hyperparameters in hyperparameter_combinations:
            file_path = (
                data_path
                / f"Lists/Vowels_a{'ll' if 'MultiChannel' in model_type.__name__ else ''}_{disease}_train.txt"
            )
            for model in training_validation(
                device=device,
                file_path=file_path,
                num_splits=num_splits,
                early_stopping_patience=early_stopping_patience,
                criterion=criterion,
                model_creator=model_creator,
                **dict(hyperparameters),
            ):
                del model

            writer.flush()
