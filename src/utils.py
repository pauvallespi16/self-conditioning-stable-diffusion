import pickle
from functools import reduce
from pathlib import Path

import torchvision.transforms as transforms

# Path constants
OUTPUT_PATH = Path("output")
IMAGES_PATH = Path("images")
CAT_IMAGES_PATH = IMAGES_PATH / "cats"
DOG_IMAGES_PATH = IMAGES_PATH / "dogs"
OUTPUT_IMAGES_PATH = IMAGES_PATH / "output"

# String constants
TWEAKING_STRING = "Tweaking model activations..."
GENERATION_STRING = "Generating activations..."


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
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def get_module_by_name(module, access_string):
    """
    Returns the module or attribute specified by the access string.

    Args:
        module: The module to start the search from.
        access_string (str): The dot-separated string representing the module or attribute to access.

    Returns:
        object: The module or attribute specified by the access string.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
