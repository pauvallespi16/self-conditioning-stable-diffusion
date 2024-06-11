import multiprocessing
import pickle
from functools import partial, reduce
from itertools import repeat, starmap
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)

# String constants
GENERATION_STRING = "Generating activations..."
EVALUATION_STRING = "Tweaking model activations..."

# VAE constants
VAE_MODEL_NAME = "stabilityai/sdxl-vae"
VAE_LAYER_NAMES = [
    # "decoder.up_blocks.0.resnets.0.conv1",
    # "decoder.up_blocks.0.resnets.0.conv2",
    # "decoder.up_blocks.3.resnets.0.conv1",
    # "decoder.up_blocks.3.resnets.0.conv2",
    "decoder.up_blocks.3.resnets.2.conv1",
    "decoder.up_blocks.3.resnets.2.conv2",
    "decoder.mid_block.resnets.1.conv1",
    "decoder.mid_block.resnets.1.conv2",
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
    if isinstance(file_path, Path) or isinstance(file_path, str):
        with open(file_path, "rb") as f:
            return pickle.load(f)


def save_pickle(data: any, file_path: Path):
    """
    Saves the given data as a pickle file.

    Args:
        data (any): The data to be saved.
        file_path (Path): The path to save the pickle file.
    """
    if file_path is not None:
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
        Dict[str, List[float]]: A dictionary of average precision for each layer.
    """
    cpus = min(cpu_count() - 1, 8) if cpus is None else cpus
    pool = Pool(processes=cpus)
    aps = {}

    for layer in tqdm(layers, desc="Computing average precision..."):
        labels = []
        responses = []

        for image_name, activations in activations_dict[layer].items():
            mean_activation = activations.mean(axis=(1, 2))
            label = 1 if "dog" in image_name else 0
            responses.append(mean_activation)
            labels.append(label)

        responses = np.column_stack(responses)
        labels = np.array(labels)
        aps[layer] = np.array(
            pool.map(partial(single_average_precision, labels=labels), responses)
        )

    pool.close()
    return aps


def compute_aggregated_activations(
    layers: List[str], activations_dict: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Aggregates activations for each layer.

    Args:
        layers (List[str]): List of layer names.
        activations_dict (Dict[str, Dict[str, List[float]]]): Dictionary containing the activations for each layer.

    Returns:
        Dict[str, Tuple[List[float], List[float]]]: Dictionary containing the aggregated activations for each layer.
    """
    results = {}
    for layer in tqdm(layers, desc="Aggregating activations..."):
        class_acts = [[], []]

        for image_name, activations in activations_dict[layer].items():
            agg_activation = np.mean(activations, axis=(1, 2))
            agg_activation = agg_activation[:, np.newaxis, np.newaxis]
            label = 1 if "dog" in image_name else 0
            class_acts[label].append(agg_activation)

        agg_class0 = np.mean(np.array(class_acts[0]), axis=0)
        agg_class1 = np.mean(np.array(class_acts[1]), axis=0)
        results[layer] = (agg_class0, agg_class1)

    return results


def compute_auroc_chunk(responses: np.ndarray, labels: List[int]) -> np.ndarray:
    """
    Function to compute Area Under the Receiver Operating Characteristic Curve (AUROC) for a chunk of data.

    Args:
        responses (np.ndarray): Array of model responses.
        labels (List[int]): Array of true labels.

    Returns:
        np.ndarray: Array of AUROC scores for the chunk of data.
    """

    # Compute and return AUROC scores for the chunk of data
    return roc_auc_score(
        labels[:, None].repeat(responses.shape[1], 1),
        responses,
        average=None,
    )


def compute_auroc(
    layers: List[str],
    activations_dict: Dict[str, Dict[str, List[float]]],
    pool: multiprocessing.Pool = None,
    chunk_size: int = 10,
) -> np.ndarray:
    """
    Compute the Area Under the Receiver Operating Characteristic (AUROC) for each layer.

    Args:
        layers (List[str]): List of layer names.
        activations_dict (Dict[str, Dict[str, List[float]]]): Dictionary containing the activations for each layer and image.
        pool (multiprocessing.Pool, optional): Pool object for parallel processing. Defaults to None.
        chunk_size (int, optional): Size of chunks to process in parallel. Defaults to 10.

    Returns:
        np.ndarray: Array containing the AUROC values for each layer.
    """
    aurocs = {}

    for layer in tqdm(layers, desc="Computing AUROC..."):
        labels = []
        responses = []

        for image_name, activations in activations_dict[layer].items():
            mean_activation = activations.mean(axis=(1, 2))
            label = 1 if "dog" in image_name else 0
            responses.append(mean_activation)
            labels.append(label)

        responses = np.array(responses)
        labels = np.array(labels)
        responses_map = [
            responses[:, start : (start + chunk_size)]
            for start in np.arange(0, responses.shape[1], chunk_size)
        ]
        args = zip(responses_map, repeat(labels))
        if pool is not None:
            ret = pool.starmap(compute_auroc_chunk, args)
        else:
            ret = starmap(compute_auroc_chunk, args)

        auroc = np.concatenate(list(ret), 0)
        auroc = np.abs(2 * auroc - 1)  # auroc es random a 0.5 per tant cal normalitzar
        aurocs[layer] = auroc

    return aurocs
