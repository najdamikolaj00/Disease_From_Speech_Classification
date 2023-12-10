from itertools import product

from torch import nn

from Evaluation.training_validation import training_validation
from Evaluation.utilities import check_cuda_availability
from Models.SpecNetModels import spec_models_specnet, get_module_name_specnet, WindowModelSpec
from config import data_path, writer

if __name__ == "__main__":
    device = check_cuda_availability()
    disease = "Rekurrensparese"
    file_path = data_path / f"Lists/Vowels_a_{disease}.txt"

    # Hyperparameters
    num_splits = 5
    early_stopping_patience = 5
    batch_size_candidates = [16, 32, 64]
    window_length_candidates = [20, 30, 40]
    window_stride_candidates = [5, 10, 15]

    criterion = nn.BCELoss()

    # Set up the model type:
    # ResNet18
    # model_type = spec_models[
    #     get_module_name("ResNet18", "Linear", "Pretrained", "Window", "MultiChannel")
    # ]

    # SpecNet
    model_type = spec_models_specnet[
        get_module_name_specnet("SpecNetWithSE", "Linear", "Continuous", "SingleChannel")
    ]

    # Start training and validation
    output_models = []

    augmentation_types = [
        "pad_zeros",
        "frequency_masking",
        "time_masking",
        "combined_masking",
        "no_augmentation",
    ]

    random_states = (7, 69, 420, 2137)

    hyperparameter_combinations = product(augmentation_types, batch_size_candidates)

    if isinstance(model_type, WindowModelSpec):
        hyperparameter_combinations = product(
            augmentation_types,
            batch_size_candidates,
            window_length_candidates,
            window_stride_candidates,
        )

    for hyperparameters in hyperparameter_combinations:
        if isinstance(model_type, WindowModelSpec):
            (
                augmentation_type,
                batch_size,
                window_length,
                window_stride,
            ) = hyperparameters
            output_models += list(
                training_validation(
                    device,
                    file_path,
                    num_splits,
                    batch_size,
                    early_stopping_patience,
                    criterion,
                    model_type,
                    augmentation=augmentation_type,
                    tun_window_size=window_length,
                    tun_window_stride=window_stride,
                )
            )
        else:
            augmentation_type, batch_size = hyperparameters
            output_models += list(
                training_validation(
                    device,
                    file_path,
                    num_splits,
                    batch_size,
                    early_stopping_patience,
                    criterion,
                    model_type,
                    augmentation=augmentation_type,
                )
            )

        writer.flush()
