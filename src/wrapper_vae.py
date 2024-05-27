from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_dataset import CustomDataset
from utils import *
from wrapper import ModelWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "stabilityai/sdxl-vae"
LAYER_NAME = "decoder.mid_block.resnets.0.conv2"

SAVE_OUTPUT_IMAGES = True
SAVE_ACTIVATIONS_FILE = OUTPUT_PATH / "dog_activations.pkl"


class VariationalAutoencoderWrapper(ModelWrapper):
    def __init__(
        self,
        model_name: str,
        device: str,
        save_output_images: bool = False,
        save_activations_file: str = None,
    ):
        super().__init__(model_name, device, save_output_images, save_activations_file)
        self.vae = AutoencoderKL.from_pretrained(self.model_name).to(self.device)
        self.vae.eval()

    def encode_img(self, input_img: torch.Tensor) -> torch.Tensor:
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        if len(input_img.shape) < 4:
            input_img = input_img.unsqueeze(0)

        with torch.no_grad():
            latents = self.vae.encode(input_img * 2 - 1)  # [0, 1] -> [-1, 1]

        return 0.18215 * latents.latent_dist.sample()  # Constant from the model

    def decode_img(self, latents: torch.Tensor) -> torch.Tensor:
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents  # Inverse of the constant from the model

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
        image = image.detach()
        return image

    def generate_activations(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        activations_dict = {}
        output_transform = transforms.ToPILImage()
        desc = GENERATION_STRING if self.save_activations_file else TWEAKING_STRING

        for image_names, batch_images in tqdm(dataloader, desc=desc):
            latents = self.encode_img(batch_images)
            reconstructed_images = self.decode_img(latents)

            if self.save_activations_file:
                activations_dict.update(
                    {
                        image_names[i]: self.activations[0][i]
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


def generate():
    dataset = CustomDataset(
        DOG_IMAGES_PATH, device=DEVICE, transform=get_transforms(), num_images=1000
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    vae = VariationalAutoencoderWrapper(
        model_name=MODEL_NAME,
        device=DEVICE,
        save_activations_file=SAVE_ACTIVATIONS_FILE,
    )

    vae.register_hook(vae.vae, LAYER_NAME)
    vae.generate_activations(dataloader)


def evaluate():
    dog_activations = load_pickle(SAVE_ACTIVATIONS_FILE)
    dataset = CustomDataset(
        CAT_IMAGES_PATH, device=DEVICE, transform=get_transforms(), num_images=10
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    vae = VariationalAutoencoderWrapper(
        model_name=MODEL_NAME,
        device=DEVICE,
        save_output_images=SAVE_OUTPUT_IMAGES,
    )

    agg_activation = np.mean(list(dog_activations.values()), axis=0)

    def hook_fn(module, input, output):
        # Replace output value by aggregation of dog activations
        median_tensor = torch.from_numpy(agg_activation).to(DEVICE)
        output = median_tensor.repeat(output.size(0), 1, 1, 1)
        noise = torch.randn_like(output) * 0.01  # Add small random noise
        output += noise
        return output

    vae.register_hook(vae.vae, LAYER_NAME, hook_fn=hook_fn)
    vae.generate_activations(dataloader)


if __name__ == "__main__":
    generate()
    evaluate()
