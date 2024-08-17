import json
import os
import pickle
from functools import reduce
from pathlib import Path
from typing import Any, List, Tuple

import torch
from PIL.Image import Image

# String constants
GENERATION_STRING = "Generating activations..."
EVALUATION_STRING = "Generating images..."

SD_LAYERS = [
    f"text_model.encoder.layers.{i}.layer_norm{j}"
    for i in range(12)
    for j in range(1, 3)
]
SD_VERSION_TO_MODEL = {
    "1.1": "CompVis/stable-diffusion-v1-1",
    "1.5": "runwayml/stable-diffusion-v1-5",
    "2.0": "stabilityai/stable-diffusion-2",
    "xl-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
}


def load_pickle(file_path: Any) -> object:
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

    return file_path


def save_pickle(data: any, file_path: Path):
    """
    Saves the given data as a pickle file.

    Args:
        data (any): The data to be saved.
        file_path (Path): The path to save the pickle file.
    """
    if file_path is not None:
        os.makedirs(file_path.parent, exist_ok=True)
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


def save_images(
    folder: Path,
    images: List[Image],
    sentences: List[str],
    image_size: Tuple[int, int] = (512, 512),
):
    """
    Saves the given images with the given name to disk.

    Args:
        folder (Path): The folder to save the images to.
        images (List[Image]): The images to save.
        sentences (List[Image]): The sentences corresponding to the images.
        image_size (Tuple[int, int], optional): The size to resize the images to. Defaults to (512, 512).
    """
    if folder is None:
        return

    for i, image in enumerate(images):
        resized_image = image.resize(image_size)
        resized_image.save(folder / f"{sentences[i]}.png")


def save_json(path: str, data: dict):
    """
    Saves the given data as a JSON file.

    Args:
        path (str): The path to save the JSON file.
        data (dict): The data to be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
