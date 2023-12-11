from itertools import count, chain
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.optim as optim
import torchaudio.transforms as T
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from Evaluation.utilities import (
    get_patients_id,
    get_files_path,
    get_patient_id,
    to_device,
)
from Models import SpectrogramDataset
from Models.Models import SpecModel, WindowModel
from config import data_path, writer, results_folder


def training_validation(
    device,
    file_path: Path,
    num_splits: int,
    batch_size: int,
    early_stopping_patience: int,
    criterion: _Loss,
    model_creator: Callable[[], SpecModel],
    augmentation="pad_zeros",
    tun_window_size=35,
    tun_window_stride=10,
    random_state=42,
):
    # Load patient IDs and file paths from a file
    patients_ids = get_patients_id(file_path)
    file_paths = get_files_path(file_path)

    # Define augmentations
    transform_no_augmentation = transforms.Compose(
        [transforms.Resize((224, 224), antialias=None)]
    )

    def add_trailing_zeros(x):
        zeros = torch.zeros((*x.shape[:-1], 500))
        zeros[:, :, : x.shape[-1]] = x
        return zeros

    transform_pad_zeros = transforms.Compose([add_trailing_zeros])

    transform_frequency_masking = transforms.Compose(
        [
            T.FrequencyMasking(freq_mask_param=50),
            transforms.Resize((224, 224), antialias=None),
        ]
    )

    transform_time_masking = transforms.Compose(
        [
            T.TimeMasking(time_mask_param=30),
            transforms.Resize((224, 224), antialias=None),
        ]
    )

    transform_combined_masking = transforms.Compose(
        [
            T.FrequencyMasking(freq_mask_param=50),
            T.TimeMasking(time_mask_param=30),
            transforms.Resize((224, 224), antialias=None),
        ]
    )

    # Choose the desired augmentation
    val_transform = transform_no_augmentation
    if augmentation == "frequency_masking":
        transform = transform_frequency_masking
    elif augmentation == "time_masking":
        transform = transform_time_masking
    elif augmentation == "combined_masking":
        transform = transform_combined_masking
    elif augmentation == "pad_zeros":
        transform = transform_pad_zeros
        val_transform = transform_pad_zeros
    else:
        transform = transform_no_augmentation

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(patients_ids, [label for _, label in patients_ids])
    ):
        best_model_weights = None
        val_losses = []

        # ResNet18 https://discuss.pytorch.org/t/altering-resnet18-for-single-channel-images/29198/6
        model_type = model_creator()
        if isinstance(model_type, WindowModel):
            model = model_type.get_model(
                device, window_size=tun_window_size, window_stride=tun_window_stride
            )
        else:
            model = model_type.get_model(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"Fold {fold + 1}/{num_splits}")

        # Get train and validation patient IDs and file paths
        train_patients = np.array(patients_ids)[train_idx]
        val_patients = np.array(patients_ids)[val_idx]

        train_files = list(
            chain.from_iterable(
                [f"{data_path}/{file}"]
                if file.endswith("0")
                else 4 * [f"{data_path}/{file}"]
                for file in file_paths
                if get_patient_id(file)[0] in train_patients[:, 0]
            )
        )
        val_files = [
            f"{data_path}/{file}"
            for file in file_paths
            if get_patient_id(file)[0] in val_patients[:, 0]
        ]

        train_dataset = SpectrogramDataset(train_files, transform)
        val_dataset = SpectrogramDataset(val_files, val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        f1_scores = []
        # Training loop
        for epoch in count():
            model.train()
            total_loss = 0.0

            for batch_idx, (inputs, labels) in tqdm(
                enumerate(train_loader, 0),
                f"Training epoch {epoch + 1}...",
                len(train_loader),
            ):
                inputs, labels = to_device(inputs, device), to_device(labels, device)
                optimizer.zero_grad()

                outputs = model(inputs)
                # print(outputs)
                target = labels.float().unsqueeze(1)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            writer.add_scalar("Loss/train", train_loss, epoch)

            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            all_labels = []
            all_predicted = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = to_device(inputs, device), to_device(
                        labels, device
                    )

                    outputs = model(inputs)

                    predicted = outputs.round().squeeze()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    labels_np = labels.cpu().numpy()
                    predicted_np = predicted.cpu().numpy()

                    if labels_np.ndim == 0:
                        labels_np = np.array([labels_np])
                    if predicted_np.ndim == 0:
                        predicted_np = np.array([predicted_np])

                    all_labels.append(labels_np)
                    all_predicted.append(predicted_np)

                    target = labels.float().unsqueeze(1)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()

                val_loss /= len(val_loader)
                writer.add_scalar("Loss/validation", val_loss, epoch)
                val_losses.append(val_loss)
                best_epoch = np.argmin(val_losses)
                if best_epoch == epoch:
                    best_model_weights = model.state_dict()
                elif epoch - best_epoch > early_stopping_patience:
                    torch.save(
                        best_model_weights,
                        results_folder.joinpath(
                            f"f1_{f1_scores[best_epoch]:.2f}_{model_type.__name__}_{augmentation}_{fold}.pth"
                        ),
                    )
                    model.load_state_dict(best_model_weights)
                    yield model
                    break

                all_labels = np.concatenate(all_labels)
                all_predicted = np.concatenate(all_predicted)

                f1 = f1_score(all_labels, all_predicted, zero_division=0.0)
                f1_scores.append(f1)
                precision = precision_score(
                    all_labels, all_predicted, zero_division=0.0
                )
                recall = recall_score(all_labels, all_predicted, zero_division=0.0)
                accuracy = correct / total

            print(
                f"Epoch [{epoch + 1}], Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}, Accuracy: {100 * accuracy:.2f}%, F1-score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
            )
