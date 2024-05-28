import os
from enum import Enum
from typing import Dict, List

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import OUTPUT_IMAGES_PATH, get_module_by_name


class Process(Enum):
    GENERATION = 1
    EVALUATION = 2


class ModelWrapper:
    """
    A wrapper class for a PyTorch model.

    Args:
        model_name (str): The name of the model.
        device (str): The device to run the model on (e.g., "cpu", "cuda").
        process (Process): The process to run (e.g., Process.GENERATION, Process.EVALUATION)
        save_output_images (bool, optional): Whether to save output images. Defaults to False.
        save_activations_file (str, optional): The file path to save the activations. Defaults to None.

    Methods:
        generation_hook: A hook function for the generation process.
        evaluation_hook: A hook function for the evaluation process.
        register_hook: Registers a hook for a specific layer and process.
        generate_activations: Generates activations for a given dataloader.
        save_reconstructed_images: Saves reconstructed images to disk.

    """

    def __init__(
        self,
        model_name: str,
        device: str,
        process: Process,
        save_output_images: bool = False,
        save_activations_file: str = None,
    ):
        self.model_name = model_name
        self.device = device
        self.process = process
        self.save_output_images = save_output_images
        self.save_activations_file = save_activations_file
        self.activations = {}
        self.module_to_name = {}
        self.process_to_hook = {
            Process.GENERATION: self.generation_hook,
            Process.EVALUATION: self.evaluation_hook,
        }
        if self.save_output_images:
            os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

    def generation_hook(self, *args):
        """
        Returns a hook function for the generation process.

        Args:
            *args: Additional arguments.

        Returns:
            Callable: The hook function.

        """
        pass

    def evaluation_hook(self, *args):
        """
        Returns a hook function for the evaluation process.

        Args:
            *args: Additional arguments.

        Returns:
            Callable: The hook function.
        """
        pass

    def register_hook(
        self,
        model: torch.nn.Module,
        layer_name: str,
        agg_activations: Dict[str, List[float]] = None,
    ):
        """
        Registers a hook for a specific layer and process.

        Args:
            model (torch.nn.Module): The model to register the hook on.
            layer_name (str): The name of the layer to register the hook on.
            agg_activations (Dict[str, List[float]], optional): The aggregated activations. Defaults to None.

        """
        hook_fn = self.process_to_hook[self.process](agg_activations)
        module = get_module_by_name(model, layer_name)
        module.register_forward_hook(hook_fn)

        if module not in self.module_to_name:
            self.module_to_name[module] = layer_name

    def generate_activations(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        """
        Generates activations for a given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to generate activations for.

        Returns:
            Dict[str, List[float]]: A dictionary containing the activations of each module.

        """
        pass

    def save_reconstructed_images(
        self,
        reconstructed_images: torch.Tensor,
        image_names: List[str],
        output_transform: transforms.Compose,
    ):
        """
        Saves reconstructed images to disk.

        Args:
            reconstructed_images (torch.Tensor): The reconstructed images.
            image_names (List[str]): The names of the images.
            output_transform (transforms.Compose): The transformation to apply to the output images.

        """
        for i, image in enumerate(reconstructed_images):
            output_image = output_transform(image)
            output_image.save(
                OUTPUT_IMAGES_PATH / f"reconstructed_{image_names[i]}.jpg"
            )
