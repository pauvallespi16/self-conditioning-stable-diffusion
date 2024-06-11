from typing import Dict, List

import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from wrapper import ModelWrapper, Process


class VariationalAutoencoderWrapper(ModelWrapper):
    """
    Wrapper class for the Variational Autoencoder model.

    Args:
        model_name (str): Name of the VAE model.
        layers (List[str]): The names of the layers to register hooks on.
        device (str): Device to run the model on.
        process (Process): The process to run (e.g., Process.GENERATION, Process.EVALUATION)
        output_images_folder (Path, optional): Folder to save output images. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        layers: List[str],
        device: str,
        process: Process,
        output_images_folder: Path = None,
    ):
        super().__init__(
            model_name=model_name,
            layers=layers,
            device=device,
            process=process,
            output_images_folder=output_images_folder,
        )
        self.vae = AutoencoderKL.from_pretrained(self.model_name).to(self.device)
        self.vae.eval()
        self.activations = {}

    def encode_img(self, input_img: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image into its latent representation.

        Args:
            input_img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Latent representation of the input image.
        """
        if len(input_img.shape) < 4:
            input_img = input_img.unsqueeze(0)

        with torch.no_grad():
            latents = self.vae.encode(input_img * 2 - 1)

        return 0.18215 * latents.latent_dist.sample()

    def decode_img(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent representation into an output image.

        Args:
            latents (torch.Tensor): Latent representation tensor.

        Returns:
            torch.Tensor: Decoded output image.
        """
        latents = (1 / 0.18215) * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach()
        return image

    def generation_hook(self, *args):
        """
        Returns a hook function for the generation process.

        Args:
            *args: Additional arguments.

        Returns:
            Callable: The hook function.

        """

        def hook_fn(module, input, output):
            module_name = self.module_to_name[module]
            if module_name not in self.activations:
                self.activations[module_name] = []

            for i in range(output.shape[0]):
                out = (
                    output[i].detach().numpy()
                    if self.device == "cpu"
                    else output[i].detach().cpu().numpy()
                )
                self.activations[module_name].append(out)

        return hook_fn

    def evaluation_hook(self, *args):
        """
        Returns a hook function for the evaluation process.

        Args:
            *args: Additional arguments.

        Returns:
            Callable: The hook function.

        """
        agg_activations = args[0]
        layer_scores = args[1]
        threshold = args[2] if len(args) > 2 else 0

        def hook_fn(module, input, output):
            # other class
            module_name = self.module_to_name[module]
            class0_agg_activations = torch.tensor(
                agg_activations[module_name][0],
                device=output.device,
                dtype=output.dtype,
            )
            # dog class
            class1_agg_activations = torch.tensor(
                agg_activations[module_name][1],
                device=output.device,
                dtype=output.dtype,
            )
            mask = (
                layer_scores[module_name] > threshold * layer_scores[module_name].max()
            )
            output[:, mask] -= class0_agg_activations[mask]
            output[:, mask] += class1_agg_activations[mask]
            return output

        return hook_fn

    def run(self, dataloader: DataLoader) -> Dict[str, Dict[str, List[float]]]:
        """
        Generates or modifies activations for each layer in the VAE model.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            Dict[str, Dict[str, List[float]]]: Dictionary containing the activations for each layer.
                The structure of the result dictionary is:
                {
                    "layer1": {
                        "image1.jpg": [np.ndarray(...), np.ndarray(...), ...],
                        "image2.jpg": [np.ndarray(...), np.ndarray(...), ...],
                        ...
                    },
                    "layer2": {
                        "image1.jpg": [np.ndarray(...), np.ndarray(...), ...],
                        "image2.jpg": [np.ndarray(...), np.ndarray(...), ...],
                        ...
                    }
                }
        """
        activations_dict = {layer: {} for layer in self.layers}
        output_transform = transforms.ToPILImage()
        desc = (
            GENERATION_STRING
            if self.process == Process.GENERATION
            else EVALUATION_STRING
        )

        for image_names, batch_images in tqdm(dataloader, desc=desc):
            latents = self.encode_img(batch_images)
            reconstructed_images = self.decode_img(latents)

            if self.process == Process.GENERATION:
                for module, activations in self.activations.items():
                    activations_dict[module].update(
                        {
                            image_names[i]: activations[i]
                            for i in range(len(image_names))
                        }
                    )

            if self.output_images_folder:
                self.save_reconstructed_images(
                    reconstructed_images, image_names, output_transform
                )

            self.activations.clear()

        return activations_dict
