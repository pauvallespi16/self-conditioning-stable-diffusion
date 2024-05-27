import os
from typing import Dict, List

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import OUTPUT_IMAGES_PATH, get_module_by_name


class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: str,
        save_output_images: bool = False,
        save_activations_file: str = None,
    ):
        self.save_output_images = save_output_images
        self.save_activations_file = save_activations_file
        self.activations = []
        if self.save_output_images:
            os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

    def register_hook(self, model: torch.nn.Module, layer_name: str):
        hook_fn = lambda module, input, output: self.activations.append(
            output.cpu().numpy()
        )
        module = get_module_by_name(model, layer_name)
        module.register_forward_hook(hook_fn)

    def generate_activations(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        pass

    def save_reconstructed_images(
        self,
        decoded_images: torch.Tensor,
        image_names: List[str],
        output_transform: transforms.Compose,
    ):
        for i, image in enumerate(decoded_images):
            output_image = output_transform(image)
            output_image.save(
                OUTPUT_IMAGES_PATH / f"reconstructed_{image_names[i]}.jpg"
            )
