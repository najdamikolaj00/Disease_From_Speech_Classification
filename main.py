from itertools import product, chain

from Evaluation.training_validation import training_validation
from Evaluation.utilities import check_cuda_availability
from Models.Models import adjust
from Config import Config, WindowParameters

if __name__ == "__main__":
    device = check_cuda_availability()

    for window_arguments, many_channels, augmentation in chain.from_iterable(
        (
            product(
                (WindowParameters(is_window=True),),
                (False, True),
                ("pad_zeros",),
            ),
            product(
                ((WindowParameters(is_window=False),),),
                (False, True),
                (
                    "frequency_masking",
                    "time_masking",
                    "combined_masking",
                    "no_augmentation",
                ),
            ),
        )
    ):
        model_creation_function = adjust(
            Config.base_model, many_channels, *window_arguments
        )
        file_path = (
            Config.lists_path
            / f'Vowels_a{"ll" if many_channels else ""}_{Config.disease}.txt'
        )
        training_validation(
            device=device,
            file_path=file_path,
            batch_size=Config.batch_size,
            num_splits=Config.num_splits,
            early_stopping_patience=Config.early_stopping_patience,
            criterion=Config.criterion,
            model_creator=model_creation_function,
            learning_rate=Config.learning_rate,
            random_state=Config.random_state,
            augmentation=augmentation,
        )
