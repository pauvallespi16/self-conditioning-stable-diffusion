import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
from utils import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "stabilityai/sdxl-vae"
LAYER_NAME = "decoder.mid_block.resnets.0.conv2"

SAVE_OUTPUT_IMAGES = False


class VariationalAutoencoderWrapper:
    def __init__(
        self,
        model_name: str,
        device: str,
        save_output_images: bool = False,
        save_activations_file: str = None,
    ):
        self.vae = AutoencoderKL.from_pretrained(model_name).to(device)
        self.vae.eval()
        self.save_output_images = save_output_images
        self.save_activations_file = save_activations_file
        self.activations = []

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

    def register_hook(self, layer_name: str):
        hook_fn = lambda module, input, output: self.activations.append(
            output.cpu().numpy()
        )
        module = get_module_by_name(self.vae, layer_name)
        module.register_forward_hook(hook_fn)

    def save_reconstructed_images(
        self,
        decoded_images: torch.Tensor,
        image_names: list,
        output_transform: transforms.Compose,
    ):
        for i, image in enumerate(decoded_images):
            output_image = output_transform(image)
            output_image.save(
                OUTPUT_IMAGES_PATH / f"reconstructed_{image_names[i]}.jpg"
            )

    def generate_activations(self, dataloader: DataLoader) -> dict:
        activations_dict = {}
        output_transform = transforms.ToPILImage()

        for image_names, batch_images in dataloader:
            latents = self.encode_img(batch_images)
            decoded_images = self.decode_img(latents)

            activations_dict.update(
                {
                    image_names[i]: self.activations[0][i]
                    for i in range(len(image_names))
                }
            )

            if SAVE_OUTPUT_IMAGES:
                self.save_reconstructed_images(
                    decoded_images, image_names, output_transform
                )

            self.activations.clear()

        if self.save_activations_file:
            save_pickle(activations_dict, self.save_activations_file)


def main():
    dataset = CustomDataset(
        DOG_IMAGES_PATH, device=DEVICE, transform=get_transforms(), num_images=5
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    vae = VariationalAutoencoderWrapper(
        model_name=MODEL_NAME,
        device=DEVICE,
        save_output_images=SAVE_OUTPUT_IMAGES,
        save_activations_file=OUTPUT_PATH / "dog_activations.pkl",
    )

    vae.register_hook(LAYER_NAME)
    vae.generate_activations(dataloader)


if __name__ == "__main__":
    main()
