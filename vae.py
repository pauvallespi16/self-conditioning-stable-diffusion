import pickle

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
from utils import DOG_IMAGES_PATH, OUTPUT_IMAGES_PATH, get_transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_OUTPUT_IMAGES = False


def encode_img(input_img: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    if len(input_img.shape) < 4:
        input_img = input_img.unsqueeze(0)

    with torch.no_grad():
        latents = vae.encode(input_img * 2 - 1)  # [0, 1] -> [-1, 1]

    return 0.18215 * latents.latent_dist.sample()  # Constant from the model


def decode_img(latents: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents  # Inverse of the constant from the model

    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
    image = image.detach()
    return image


def main():
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.eval()

    input_transform, output_transform = get_transforms()
    dataset = CustomDataset(DOG_IMAGES_PATH, device=device, transform=input_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    activations = []

    def hook_fn(module, input, output):
        # output is shape 8, 3, 224, 224
        activations.append(output)

    vae.decoder.mid_block.resnets[0].conv2.register_forward_hook(hook_fn)

    activations_dict = {}
    for image_names, batch_images in dataloader:
        latents = encode_img(batch_images, vae)
        decoded_images = decode_img(latents, vae)

        activations_dict.update(
            {image_names[i]: activations[0][i] for i in range(len(image_names))}
        )

        if SAVE_OUTPUT_IMAGES:
            for i, image in enumerate(decoded_images):
                output_image = output_transform(image)
                output_image.save(
                    OUTPUT_IMAGES_PATH / f"reconstructed_{image_names[i]}.jpg"
                )

        activations.clear()

    with open("output/activations.pkl", "wb") as f:
        pickle.dump(activations_dict, f)


if __name__ == "__main__":
    main()
