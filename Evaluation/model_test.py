from pathlib import Path
from statistics import mean

import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from Evaluation.utilities import (
    get_files_path,
    to_device,
)
from Models import SpectrogramDataset
from config import data_path


def test_model(device, file_name, model, batch_size: int = 1):
    for mask in ('frequency_mask', 'time_masking', 'combined_masking', 'no_augmentation'):
        print(model.__name__, mask)
        accuracies, f1s, precisions, recalls = [], [], [], []
        for model_weighs in Path(f'{data_path}/results').glob(f'*{model.__name__}_{mask}*'):
            model.load_state_dict(torch.load(model_weighs, map_location=device))
            test_files = list(f"{data_path}/{file}" for file in get_files_path(file_name))

            transform = transforms.Compose([transforms.Resize((224, 224), antialias=None)])
            test_dataset = SpectrogramDataset(test_files, transform)
            test_loader = DataLoader(
                test_dataset, shuffle=False, pin_memory=True, batch_size=batch_size
            )

            model.eval()
            model.to(device)  # Move model to device

            correct = 0
            total = 0

            all_labels = []
            all_predicted = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = to_device(inputs, device), to_device(labels, device)

                    outputs = model(inputs)
                    predicted = outputs.round().reshape(batch_size)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu())
                    all_predicted.extend(predicted.cpu())

            accuracy = correct / total
            f1 = f1_score(all_labels, all_predicted, zero_division=0.0)
            precision = precision_score(all_labels, all_predicted, zero_division=0.0)
            recall = recall_score(all_labels, all_predicted, zero_division=0.0)
            accuracies.append(accuracy)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        print(f"Test Accuracy: {100 * mean(accuracies):.2f}%")
        print(f"F1-score: {mean(f1s):.2f}, Precision: {mean(precisions):.2f}, Recall: {mean(recalls):.2f}")
