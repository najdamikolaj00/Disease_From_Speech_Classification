import os
import numpy as np
import librosa
import librosa.display
import matplotlib
from PIL import Image

matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch as tc
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """
    A custom PyTorch dataset for handling audio spectrogram data.
    """

    def __init__(
        self,
        paths_to_audio,
        transform=None,
        hop_length=512,
        n_fft=2048,
        n_mels=128,
        fmin=0,
        fmax=None,
    ):
        """
        Initializes the dataset.

        Args:
            paths_to_audio (list): List of file paths to audio files.
            transform (callable): A function/transform to apply to the spectrogram data.
            hop_length (int): Number of samples between successive frames.
            n_fft (int): Number of samples in each window.
            n_mels (int): Number of mel filterbanks.
            fmin (float): Minimum frequency.
            fmax (float): Maximum frequency.
        """
        self.paths_to_audio = paths_to_audio
        self.transform = transform
        self.samples = {}  # Dictionary to store sample information
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Extract and store sample information
        for path in self.paths_to_audio:
            audio_path, label = path.split(" ")
            sample_id = os.path.splitext(os.path.basename(audio_path))[0]
            sample_id = sample_id.split("_")[0]

            if sample_id not in self.samples:
                self.samples[sample_id] = {
                    "audio_path": audio_path,
                    "label": int(label),
                }
            else:
                pass

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Gets an item (spectrogram and label) from the dataset.

        Args:
            idx (int): Index of the sample in the dataset.

        Returns:
            torch.Tensor: Spectrogram tensor.
            int: Sample label.
        """
        sample_id = list(self.samples.keys())[idx]
        audio_path = self.samples[sample_id]["audio_path"]

        y, sr = librosa.load(audio_path, sr=None)

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        #pil_img = Image.fromarray(log_mel_spec_db)

        #log_mel_spec_db = np.array(log_mel_spec_db)

        if self.transform:
            # spectrogram = tc.tensor(log_mel_spec_db, dtype=tc.float32)
            # spectrogram = self.transform(spectrogram)
            pil_img = Image.fromarray(log_mel_spec_db)

            pil_img = self.transform(pil_img)

            log_mel_spec_db = np.array(pil_img)

        #spectrogram = tc.tensor(log_mel_spec_db, dtype=tc.float32)

        label = self.samples[sample_id]['label']

        return log_mel_spec_db, label


if __name__ == "__main__":
    dataset = SpectrogramDataset(["Data/Vowels/Dysphonie/368_a.wav 1"])
    spectrogram, label = dataset[0]
    librosa.display.specshow(spectrogram.numpy(), cmap="plasma")
    # plt.savefig('spec.jpg')
    # plt.show()
