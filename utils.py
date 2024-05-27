from pathlib import Path
from typing import Tuple

import torchvision.transforms as transforms

IMAGES_PATH = Path("images")

CAT_IMAGES_PATH = IMAGES_PATH / "cats"
DOG_IMAGES_PATH = IMAGES_PATH / "dogs"

OUTPUT_IMAGES_PATH = IMAGES_PATH / "output"


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    input_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # If batch, this is necessary
            transforms.ToTensor(),
        ]
    )
    output_transform = transforms.ToPILImage()
    return input_transform, output_transform
