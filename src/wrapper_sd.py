from typing import Dict, List

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
from utils import *
from wrapper import ModelWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "..."
LAYER_NAME = "..."
SAVE_OUTPUT_IMAGES = True


class StableDifussionWrapper(ModelWrapper):
    def __init__(
        self,
        model_name: str,
        device: str,
        save_output_images: bool = False,
        save_activations_file: str = None,
    ):
        super().__init__(model_name, device, save_output_images, save_activations_file)
        self.sd = None # ...
        self.sd.eval()

    def generate_activations(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        activations_dict = {}
        output_transform = transforms.ToPILImage()

        for image_names, batch_images in dataloader:
            reconstructed_images = self.sd(batch_images)  # ...
            # ...

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


def main():
    dataset = CustomDataset(
        DOG_IMAGES_PATH, device=DEVICE, transform=get_transforms(), num_images=5
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    sd = StableDifussionWrapper(
        model_name=MODEL_NAME,
        device=DEVICE,
        save_output_images=SAVE_OUTPUT_IMAGES,
        save_activations_file=OUTPUT_PATH / "dog_activations.pkl",
    )

    sd.register_hook(sd.sd, LAYER_NAME)
    sd.generate_activations(dataloader)


if __name__ == "__main__":
    main()
