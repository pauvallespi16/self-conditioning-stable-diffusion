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

SAVE_OUTPUT_IMAGES = True
SAVE_ACTIVATIONS_FILE = OUTPUT_PATH / "dog_activations.pkl"


class VariationalAutoencoderWrapper(ModelWrapper):
    """
    Wrapper class for the Variational Autoencoder model.

    Args:
        model_name (str): Name of the VAE model.
        layers (List[str]): The names of the layers to register hooks on.
        device (str): Device to run the model on.
        process (Process): The process to run (e.g., Process.GENERATION, Process.EVALUATION)
        save_output_images (bool, optional): Whether to save the output images. Defaults to False.
        save_activations_file (str, optional): File path to save the activations. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        layers: List[str],
        device: str,
        process: Process,
        save_output_images: bool = False,
        save_activations_file: str = None,
    ):
        super().__init__(
            model_name=model_name,
            layers=layers,
            device=device,
            process=process,
            save_output_images=save_output_images,
            save_activations_file=save_activations_file,
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

        def hook_fn(module, input, output):
            module = self.module_to_name[module]
            if module not in self.activations:
                self.activations[module] = []

            for i in range(output.shape[0]):
                out = (
                    output[i].detach().numpy()
                    if self.device == "cpu"
                    else output[i].detach().cpu().numpy()
                )
                self.activations[module].append(out)

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
        units = args[1] if len(args) > 1 and args[1] else slice(None)

        def hook_fn(module, input, output):
            module_name = self.module_to_name[module]
            agg_activations_tensor = torch.from_numpy(agg_activations[module_name])
            agg_activations_tensor = agg_activations_tensor.to(self.device)
            output[:, :, :, units] = agg_activations_tensor[:, :, units]
            return output

        return hook_fn

    def run(self, dataloader: DataLoader) -> Dict[str, Dict[str, List[float]]]:
        """
        Generates or modifies activations for each layer in the VAE model.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            Dict[str, Dict[str, List[float]]]: Dictionary containing the activations for each layer.
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
        activations_dict = {layer: {} for layer in self.layers}
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


def generate() -> Dict[str, Dict[str, List[float]]]:
    """
    Generate activations for the VAE model.
    """
    dataset = CustomDataset(
        DOG_IMAGES_PATH, device=DEVICE, transform=get_transforms(), num_images=200
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    vae = VariationalAutoencoderWrapper(
        model_name=VAE_MODEL_NAME,
        layers=VAE_LAYER_NAMES,
        device=DEVICE,
        process=Process.GENERATION,
        # save_activations_file=SAVE_ACTIVATIONS_FILE,
    )

    vae.register_hooks(vae.vae)
    activations = vae.run(dataloader)

    return activations


def evaluate(activations: Dict[str, Dict[str, List[float]]] = None):
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
        model_name=VAE_MODEL_NAME,
        layers=VAE_LAYER_NAMES,
        device=DEVICE,
        process=Process.EVALUATION,
        save_output_images=SAVE_OUTPUT_IMAGES,
    )
    
    agg_activations = {
        module: np.median(list(acts.values()), axis=0)
        for module, acts in tqdm(activations.items(), desc="Aggregating activations")
    }
    units = slice(0, 14)

    vae.register_hooks(vae.vae, agg_activations)
    vae.run(dataloader)


if __name__ == "__main__":
    dog_activations = generate()
    evaluate(dog_activations)
