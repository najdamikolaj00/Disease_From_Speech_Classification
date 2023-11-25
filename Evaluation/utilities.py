"""File contains additional utilities"""
import os
from itertools import starmap
from pathlib import Path

import torch as tc


# Function to check CUDA availability
def check_cuda_availability():
    """
    Checks if a CUDA-compatible GPU is available.

    Returns:
        tc.device: A PyTorch device object set to "cuda" if CUDA is available,
        or "cpu" if not.
    """
    return tc.device("cuda" if tc.cuda.is_available() else "cpu")


# Function to move data to a specified device
def to_device(data, device):
    """
    Moves PyTorch tensors or data structures to a specified device.

    Args:
        data: Input data to move to the device.
        device: The target device ("cuda" for GPU or "cpu" for CPU).

    Returns:
        data: Input data moved to the specified device.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Function to get file paths from a text file
def get_files_path(file_name):
    """
    Reads a text file containing a list of file paths and returns a list of
    these file paths.

    Args:
        file_name (str): The name of the text file containing file paths.

    Returns:
        list: A list of file paths, with each line of the file as an element
        in the list.
    """
    return list(
        line for line in Path(file_name).read_text().splitlines()
    )


# Function to extract patient ID from a file path
def get_patient_id(file):
    """
    Extracts the patient ID from a file path.

    Args:
        file (str): The file path.

    Returns:
        str: The extracted patient ID.
    """
    spec_path, label = file.split(" ")
    patient_id = os.path.splitext(os.path.basename(spec_path))[0]
    patient_id = patient_id.split("_")[0]
    return patient_id


# Function to get a list of unique patient IDs from a text file
def get_patients_id(file_name):
    """
    Reads a text file containing file paths and extracts a list of unique
    patient IDs.

    Args:
        file_name (str): The name of the text file containing file paths.

    Returns:
        list: A list of unique patient IDs.
    """
    return list(set(map(get_patient_id, Path(file_name).read_text().splitlines())))


# Function to save results to a text file
def save_results(output_file, metrics):
    """
    Saves metrics (e.g., Mean Accuracy, Mean F1 Score) to a text file.

    Args:
        output_file (str): The name of the output text file.
        metrics (list of tuples): A list of metric tuples, each containing
        (metric_name, mean, std).
    """
    Path(output_file).write_text(
        "\n".join(starmap("{}: {:.2f} (±{:.2f})".format, metrics))
    )