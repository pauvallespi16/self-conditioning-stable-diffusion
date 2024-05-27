from pathlib import Path
from typing import Union

import torch
import torchvision.transforms as transforms
from PIL import Image

IMAGES_PATH = Path("images")

INPUT_IMAGES_PATH = IMAGES_PATH / "input"
OUTPUT_IMAGES_PATH = IMAGES_PATH / "output"

CAT_IMAGES_PATH = INPUT_IMAGES_PATH / "cats"
DOG_IMAGES_PATH = INPUT_IMAGES_PATH / "dogs"


def load_image(
    image_path: Path, transforms: transforms = None
) -> Union[Image.Image, torch.Tensor]:
    image = Image.open(image_path)

    if transforms:
        image = transforms(image)

    return image
