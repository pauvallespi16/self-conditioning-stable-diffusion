import pickle
from functools import partial, reduce
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# String constants
TWEAKING_STRING = "Tweaking model activations..."
GENERATION_STRING = "Generating activations..."

# VAE constants
VAE_MODEL_NAME = "stabilityai/sdxl-vae"
VAE_LAYER_NAMES = [
    "decoder.mid_block.resnets.0.conv1",
    # "decoder.mid_block.resnets.0.conv2",
    # "decoder.up_blocks.0.resnets.0.conv1",
    # "decoder.up_blocks.0.resnets.0.conv2",
    # "decoder.up_blocks.1.resnets.1.conv1",
    # "decoder.up_blocks.1.resnets.1.conv2",
]

# SD constants
SD_MODEL_NAME = "CompVis/stable-diffusion-v1-1"
SD_LAYER_NAMES = ["..."]


def get_transforms() -> transforms.Compose:
    """
    Returns a torchvision.transforms.Compose object that represents a series of image transformations.

    Returns:
        transforms.Compose: A Compose object that applies a series of transformations to an image.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),  # If batch, this is necessary
            transforms.ToTensor(),
        ]
    )


def load_pickle(file_path: Path) -> object:
    """
    Loads and returns the object stored in a pickle file.

    Args:
        file_path (Path): The path to the pickle file.

    Returns:
        object: The object stored in the pickle file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: any, file_path: Path):
    """
    Saves the given data as a pickle file.

    Args:
        data (any): The data to be saved.
        file_path (Path): The path to save the pickle file.
    """
    if file_path:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)


def get_module_by_name(module: torch.nn.Module, access_string: str) -> torch.nn.Module:
    """
    Returns the module or attribute specified by the access string.

    Args:
        module: The module to start the search from.
        access_string (str): The dot-separated string representing the module or attribute to access.

    Returns:
        torch.nn.Module: The module or attribute specified by the access string.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def normalize(array):
    """
    Normalizes the given array.

    Args:
        array: The array to be normalized.

    Returns:
        The normalized array.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def single_average_precision(unit_response: np.ndarray, labels: List[int]) -> float:
    """
    Computes the average precision score for a single unit response.

    Args:
        unit_response (np.ndarray): The unit response.
        labels (List[int]): The corresponding labels.

    Returns:
        float: The average precision score.
    """
    return average_precision_score(y_true=labels, y_score=unit_response)


def compute_average_precision(
    layers: List[str],
    activations_dict: Dict[str, Dict[str, List[float]]],
    cpus: int = None,
) -> Dict[str, List[float]]:
    """
    Computes the average precision scores for the given layers and activations.

    Args:
        layers (List[str]): The list of layer names.
        activations_dict (Dict[str, Dict[str, List[float]]]): The dictionary of activations for each layer and image.
        cpus (int, optional): The number of CPUs to use for parallel processing. Defaults to None.

    Returns:
        Dict[str, List[float]]: A dictionary of average precision scores for each layer.
    """
    cpus = min(cpu_count() - 1, 8) if cpus is None else cpus
    pool = Pool(processes=cpus)
    aps = {}

    for layer in tqdm(layers, desc="Computing average precision scores..."):
        shape = None
        labels = []
        responses = []

        for image_name, activations in activations_dict[layer].items():
            if not shape:
                shape = activations.shape
            activations = normalize(activations.flatten())
            label = 1 if "dog" in image_name else 0
            responses.append(activations)
            labels.append(label)

        responses = np.column_stack(responses)
        labels = np.array(labels)

        aps[layer] = np.array(
            pool.map(partial(single_average_precision, labels=labels), responses)
        ).reshape(shape)

    pool.close()
    return aps
