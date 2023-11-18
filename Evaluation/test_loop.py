import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import f1_score, precision_score, recall_score

from Models import SpectrogramDataset, SpecNet, SpecNetWithAttention, SpecNetWithSE
from Evaluation.utilities import (
    get_patients_id,
    get_files_path,
    to_device,
    check_cuda_availability,
)


def test_model(device, file_name, model, criterion):
    # Load test patient IDs and file paths
    test_patients = get_patients_id(file_name)
    test_files = get_files_path(file_name)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    # Create test dataset and data loader
    test_dataset = SpectrogramDataset(test_files, transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model.eval()
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

            target = labels
            num_classes = 2
            target_one_hot = torch.zeros(target.size(0), num_classes, device=device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)

            loss = criterion(outputs, target_one_hot.float())
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted)
    recall = recall_score(all_labels, all_predicted)

    print(f"Test Accuracy: {100 * accuracy:.2f}%")
    print(f"F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
    print(f"Average Test Loss: {average_test_loss:.4f}")


if __name__ == "__main__":
    device = check_cuda_availability()
    # Define the file name containing data paths

    file_name = "Data/Lists/Vowels_a_Dysphonie.txt"
    model = SpecNet().to(device)
    criterion = nn.BCELoss()
    test_model(device, file_name, model, criterion)
