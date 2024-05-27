from typing import Tuple

import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL

from utils import DOG_IMAGES_PATH, OUTPUT_IMAGES_PATH, load_image

device = "cuda" if torch.cuda.is_available() else "cpu"


def encode_img(input_img: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    if len(input_img.shape) < 4:
        input_img = input_img.unsqueeze(0)

    with torch.no_grad():
        latents = vae.encode(input_img * 2 - 1)  # [0, 1] -> [-1, 1]

    return 0.18215 * latents.latent_dist.sample() # Constant from the model


def decode_img(latents: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents # Inverse of the constant from the model

    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1) # [-1, 1] -> [0, 1]
    image = image.detach()
    return image


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    input_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),  # If batch, this is necessary
            transforms.ToTensor(),
        ]
    )
    output_transform = transforms.ToPILImage()
    return input_transform, output_transform


def main():
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.eval()

    image_names = ["dog.1.jpg"]
    input_transform, output_transform = get_transforms()
    images = [
        load_image(DOG_IMAGES_PATH / name, input_transform) 
        for name in image_names
    ]
    batch_images = torch.stack(images)

    activations = []
    def hook_fn(module, input, output):
        activations.append(output)

    vae.decoder.mid_block.resnets[0].conv2.register_forward_hook(hook_fn)
    
    latents = encode_img(batch_images, vae)
    decoded_images = decode_img(latents, vae)

    print(activations[0].shape)
    # output_image = output_transform(decoded_images[0])
    # output_image.save(OUTPUT_IMAGES_PATH / f"reconstructed_{image_names[0]}")


if __name__ == "__main__":
    main()
