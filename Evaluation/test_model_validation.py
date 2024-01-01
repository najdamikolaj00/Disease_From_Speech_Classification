import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from Evaluation.utilities import (
    get_patients_id,
    get_files_path,
    to_device,
)
from Models import SpectrogramDataset


def test_model(device, file_name, model, criterion, batch_size=8):
    test_patients = get_patients_id(file_name)
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


    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    print(f'F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
    print(f'Average Test Loss: {average_test_loss:.4f}')

