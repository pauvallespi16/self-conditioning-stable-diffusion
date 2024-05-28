from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_dataset import CustomDataset
from utils import *
from wrapper import ModelWrapper, Process

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "stabilityai/sdxl-vae"
LAYER_NAMES = [
    "decoder.mid_block.resnets.0.conv1",
    "decoder.mid_block.resnets.0.conv2",
]

SAVE_OUTPUT_IMAGES = False
SAVE_ACTIVATIONS_FILE = OUTPUT_PATH / "dog_activations.pkl"


class VariationalAutoencoderWrapper(ModelWrapper):
    """
    Wrapper class for the Variational Autoencoder model.

    Args:
        model_name (str): Name of the VAE model.
        device (str): Device to run the model on.
        process (Process): The process to run (e.g., Process.GENERATION, Process.EVALUATION)
        save_output_images (bool, optional): Whether to save the output images. Defaults to False.
        save_activations_file (str, optional): File path to save the activations. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        process: Process,
        save_output_images: bool = False,
        save_activations_file: str = None,
    ):
        super().__init__(
            model_name, device, process, save_output_images, save_activations_file
        )
        self.vae = AutoencoderKL.from_pretrained(self.model_name).to(self.device)
        self.vae.eval()

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

        def hook_fn(m, i, o):
            module = self.module_to_name[m]
            if module not in self.activations:
                self.activations[module] = []

            for i in range(o.shape[0]):
                output = (
                    o[i].detach().numpy()
                    if self.device == "cpu"
                    else o[i].detach().cpu().numpy()
                )
                self.activations[module].append(output)

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

        def hook_fn(m, i, o):
            agg_activations_tensor = torch.from_numpy(
                agg_activations[self.module_to_name[m]]
            ).to(self.device)
            output = agg_activations_tensor.repeat(o.size(0), 1, 1, 1)
            noise = torch.randn_like(output) * 100  # Add small random noise
            output += noise
            return output

        return hook_fn

    def generate_activations(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        """
        Generate activations for each layer in the VAE model.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            Dict[str, List[float]]: Dictionary containing the activations for each layer.
                The struct of the result dictionary is:
                    activations_dict = {
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

        """
        activations_dict = {layer: {} for layer in LAYER_NAMES}
        output_transform = transforms.ToPILImage()
        desc = (
            GENERATION_STRING if self.process == Process.GENERATION else TWEAKING_STRING
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

            if self.save_output_images:
                self.save_reconstructed_images(
                    reconstructed_images, image_names, output_transform
                )

            self.activations.clear()

        if self.save_activations_file:
            save_pickle(activations_dict, self.save_activations_file)

        return activations_dict


def generate():
    """
    Generate activations for the VAE model.
    """
    dataset = CustomDataset(
        DOG_IMAGES_PATH, device=DEVICE, transform=get_transforms(), num_images=500
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    vae = VariationalAutoencoderWrapper(
        model_name=MODEL_NAME,
        device=DEVICE,
        process=Process.GENERATION,
        # save_activations_file=SAVE_ACTIVATIONS_FILE,
    )

    for layer in LAYER_NAMES:
        vae.register_hook(vae.vae, layer)
    activations = vae.generate_activations(dataloader)

    return activations


def evaluate(activations: Dict[str, List[float]] = None):
    """
    Evaluate the VAE model using pre-generated activations.
    """
    if activations is None:
        activations = load_pickle(SAVE_ACTIVATIONS_FILE)

    dataset = CustomDataset(
        CAT_IMAGES_PATH, device=DEVICE, transform=get_transforms(), num_images=10
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    vae = VariationalAutoencoderWrapper(
        model_name=MODEL_NAME,
        device=DEVICE,
        process=Process.EVALUATION,
        save_output_images=SAVE_OUTPUT_IMAGES,
    )

    agg_activations = {
        module: np.median(list(acts.values()), axis=0)
        for module, acts in activations.items()
    }

    for layer in LAYER_NAMES:
        vae.register_hook(vae.vae, layer, agg_activations=agg_activations)
    vae.generate_activations(dataloader)


if __name__ == "__main__":
    dog_activations = generate()
    evaluate(dog_activations)
