import os
import numpy as np
from PIL import Image

import torch as tc
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    """
    A custom PyTorch dataset for handling spectrogram data.
    """

    def __init__(self, paths_to_spectrograms, transform):
        """
        Initializes the dataset.

        Args:
            paths_to_spectrograms (list): List of file paths to spectrogram images.
            transform (callable): A function/transform to apply to the spectrogram data.
        """
        self.paths_to_spectrograms = paths_to_spectrograms
        self.transform = transform
        self.patient_spectrograms = {}
        self.patient_labels = {}

        # Organize data by patient ID and extract labels
        for path in paths_to_spectrograms:
            spec_path, label = path.split(' ')
            patient_id = os.path.splitext(os.path.basename(spec_path))[0]
            patient_id = patient_id.split('_')[0]

            if patient_id not in self.patient_labels:
                self.patient_labels[patient_id] = int(label)

            if patient_id not in self.patient_spectrograms:
                self.patient_spectrograms[patient_id] = []
            self.patient_spectrograms[patient_id].append(spec_path)

    def __len__(self):
        """
        Returns the number of patients in the dataset.

        Returns:
            int: The number of patients.
        """
        return len(self.patient_spectrograms)

    def __getitem__(self, idx):
        """
        Gets an item (spectrogram and label) from the dataset.

        Args:
            idx (int): Index of the patient in the dataset.

        Returns:
            torch.Tensor: Combined spectrogram tensor.
            int: Patient label.
        """
        patient_id = list(self.patient_spectrograms.keys())[idx]
        spectrogram_paths = self.patient_spectrograms[patient_id]

        # Load and preprocess spectrograms
        spectrograms = [Image.open(path).convert('L') for path in spectrogram_paths]  # Convert to grayscale

        if self.transform:
            spectrograms = [self.transform(img) for img in spectrograms]

        spectrograms = [tc.tensor(np.array(img), dtype=tc.float32) for img in spectrograms]
        spectrograms = [img / 255.0 for img in spectrograms]

        combined_spectrogram = tc.stack(spectrograms, dim=0)

        label = self.patient_labels[patient_id]

        return combined_spectrogram, label
