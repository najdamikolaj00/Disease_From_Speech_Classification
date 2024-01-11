import re
from itertools import chain
from pathlib import Path
from typing import Literal

from Config import Config


def create_single_vowel_set(
    rek_data_path: Path, health_data_path: Path, vowel: Literal["a", "i", "u"]
):
    (Config.lists_path / f"Vowels_{vowel}_{Config.disease}.txt").write_text(
        "\n".join(
            chain.from_iterable(
                (
                    (
                        f"Vowels/{Config.disease}/{path.name} 1"
                        for path in rek_data_path.glob(f"*_{vowel}.wav")
                    ),
                    (
                        f"Vowels/Healthy/{path.name} 0"
                        for path in health_data_path.glob(f"*_{vowel}.wav")
                    ),
                )
            )
        )
    )


def create_multi_vowel_set(diseased_data_path: Path, health_data_path: Path):
    all_patients_disease = tuple(
        set(
            chain.from_iterable(
                map(re.compile(r"\d+").findall, map(str, diseased_data_path.iterdir()))
            )
        )
    )
    train_recordings_disease = tuple(
        filter(
            lambda path: any(
                map(re.findall(r"\d+", path.name)[0].__eq__, all_patients_disease)
            ),
            diseased_data_path.iterdir(),
        )
    )
    all_patients_healthy = tuple(
        set(
            chain.from_iterable(
                map(re.compile(r"\d+").findall, map(str, health_data_path.iterdir()))
            )
        )
    )
    train_recordings_healthy = tuple(
        filter(
            lambda path: any(
                map(re.findall(r"\d+", path.name)[0].__eq__, all_patients_healthy)
            ),
            health_data_path.iterdir(),
        )
    )
    (Config.lists_path / f"Vowels_all_{Config.disease}.txt").write_text(
        "\n".join(
            chain.from_iterable(
                (
                    (
                        f"Vowels/{Config.disease}/{path.name} 1"
                        for path in train_recordings_disease
                    ),
                    (
                        f"Vowels/Healthy/{path.name} 0"
                        for path in train_recordings_healthy
                    ),
                )
            )
        )
    )


def main():
    rek_data_path = Config.root_path / f"Data/Vowels/{Config.disease}"
    health_data_path = Path("Data/Vowels/Healthy")
    for vowel in ("a", "i", "u"):
        create_single_vowel_set(rek_data_path, health_data_path, vowel)
    create_multi_vowel_set(rek_data_path, health_data_path)


if __name__ == "__main__":
    main()
