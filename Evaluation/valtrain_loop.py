import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

# Import utility functions and classes from local modules
from utilities import (
    check_cuda_availability,
    to_device,
    get_files_path,
    get_patient_id,
    get_patients_id,
    save_results
)

# Import model, dataset, and SEBlock from local modules
from Models.SpecNet import SpecNet
from Models.Dataset import SpectrogramDataset

def training_validation(device, file_name, num_epochs, num_splits, batch_size, model, criterion, optimizer):
    # Load patient IDs and file paths from a file
    patients_ids = get_patients_id(file_name)
    file_paths = get_files_path(file_name)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Lists to store evaluation metrics
    acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # Initialize K-fold cross-validation
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(patients_ids)):
        print(f"Fold {fold + 1}/{num_splits}")

        # Get train and validation patient IDs and file paths
        train_patients = np.array(patients_ids)[train_idx]
        val_patients = np.array(patients_ids)[val_idx]

        train_files = [file for file in file_paths if get_patient_id(file) in train_patients]
        val_files = [file for file in file_paths if get_patient_id(file) in val_patients]

        # Create training and validation datasets and data loaders
        train_dataset = SpectrogramDataset(train_files, transform)
        val_dataset = SpectrogramDataset(val_files, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = to_device(inputs, device), to_device(labels, device)

                optimizer.zero_grad()

                outputs = model(inputs)
                target = labels
                num_classes = 2

                target_one_hot = torch.zeros(target.size(0), num_classes, device=device)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)

                loss = criterion(outputs, target_one_hot.float())

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            train_losses.append(train_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_loss}")

        # Validation loop
        model.eval()
        correct = 0
        total = 0

        val_loss = 0

        all_labels = []
        all_predicted = []

        with torch.no_grad():
            for data in val_loader:
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
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_loader)

        val_losses.append(average_val_loss)

        # Compute evaluation metrics (accuracy, F1-score, precision, recall)
        f1 = f1_score(all_labels, all_predicted)
        precision = precision_score(all_labels, all_predicted)
        recall = recall_score(all_labels, all_predicted)
        accuracy = correct / total

        f1_2 = f1_score(labels.cpu(), predicted.cpu())
        precision_2 = precision_score(labels.cpu(), predicted.cpu())
        recall_2 = recall_score(labels.cpu(), predicted.cpu())

        print(f'Fold {fold + 1} Accuracy: {100 * correct / total:.2f}%, F1-score: {f1_2}, Precision: {precision_2}, Recall: {recall_2}')

        acc_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Compute mean and standard deviation of evaluation metrics
    mean_acc = sum(acc_scores) / len(acc_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)

    std_acc = np.std(acc_scores)
    std_f1 = np.std(f1_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)

    # Save results to an output file
    metrics = [
        ("Mean Accuracy", mean_acc, std_acc),
        ("Mean F1 Score", mean_f1, std_f1),
        ("Mean Precision", mean_precision, std_precision),
        ("Mean Recall", mean_recall, std_recall)
    ]

    save_results('output.txt', metrics)

    print(f'Mean Accuracy: {mean_acc:.2f} (±{std_acc:.2f})')
    print(f'Mean F1 Score: {mean_f1:.2f} (±{std_f1:.2f})')
    print(f'Mean Precision: {mean_precision:.2f} (±{std_precision:.2f})')
    print(f'Mean Recall: {mean_recall:.2f} (±{std_recall:.2f})')

if __name__ == '__main__':
    # Check GPU availability
    device = check_cuda_availability()
    # Define the file name containing data paths
    file_name = 'Data/Lists/Vowels_a_Dysphonie.txt'

    # Hyperparameters
    num_epochs = 50
    num_splits = 5 
    batch_size = 32

    # Initialize the model, loss criterion, and optimizer
    model = SpecNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training and validation
    training_validation(device, file_name, num_epochs, num_splits, batch_size, model, criterion, optimizer)
