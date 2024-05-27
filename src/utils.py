import pickle
from functools import reduce
from pathlib import Path
from typing import Tuple

import torchvision.transforms as transforms

OUTPUT_PATH = Path("output")
IMAGES_PATH = Path("images")

CAT_IMAGES_PATH = IMAGES_PATH / "cats"
DOG_IMAGES_PATH = IMAGES_PATH / "dogs"

OUTPUT_IMAGES_PATH = IMAGES_PATH / "output"


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),  # If batch, this is necessary
            transforms.ToTensor(),
        ]
    )


def load_pickle(file_path: Path) -> object:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: any, file_path: Path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
