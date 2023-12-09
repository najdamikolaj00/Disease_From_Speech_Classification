from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Evaluation.utilities import (
    get_files_path,
    to_device,
    check_cuda_availability,
)
from Models import SpectrogramDataset

root_path = Path(".")
data_path = root_path / "Data"
session_time = datetime.now().strftime("%Y%m%d%H%M")
results_folder = data_path.joinpath(f"results/{session_time}")
results_folder.mkdir(exist_ok=True, parents=True)
summary_folder = data_path.joinpath(f"summaries/{session_time}")
summary_folder.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(str(summary_folder))


def test_model(device, file_name, model_weights_path: Path, criterion, batch_size=8):
    test_files = get_files_path(file_name)

    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=None)
    ])
    test_dataset = SpectrogramDataset(test_files, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    model.to(device)  # Move model to device

    test_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = to_device(inputs, device), to_device(labels, device)

            outputs = model(inputs)
            predicted = outputs.round().squeeze()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

            target = labels.float().unsqueeze(1)
            loss = criterion(outputs, target)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predicted, zero_division=0.0)
    precision = precision_score(all_labels, all_predicted, zero_division=0.0)
    recall = recall_score(all_labels, all_predicted, zero_division=0.0)

    output_file.write(f'Test Accuracy: {100 * accuracy:.2f}%\n')
    output_file.write(f'F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}\n')
    output_file.write(f'Average Test Loss: {average_test_loss:.4f}\n')
    output_file.flush()
    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    print(f'F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
    print(f'Average Test Loss: {average_test_loss:.4f}')


if __name__ == '__main__':
    device = check_cuda_availability()
    disease = 'Rekurrensparese'
    file_name = f'/content/drive/MyDrive/Deep_Learing_Course_Winter_2023/Data/Lists/Lists_colab/Vowels_a_{disease}_test.txt'

    test_model(device, file_name, model, nn.BCELoss(), batch_size=8)
