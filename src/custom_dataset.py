import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

random.seed(42)


class CustomDataset(Dataset):
    """
    A custom dataset class for loading and processing images.

    Args:
        img_dir (Path): The directory containing the image files.
        device (str): The device to use for image processing.
        transform (transforms.Compose, optional): A transform to apply to the images. Default is None.
        num_images (int, optional): The number of images to load. Default is 100. If -1 is pased, all images are used.

    """

    def __init__(
        self,
        img_dir: Path,
        device: str,
        transform: transforms.Compose = None,
        num_images: int = None,
    ):
        images = list(img_dir.glob("*"))
        if not num_images:
            num_images = len(images)
        self.img_paths = images[:num_images]
        self.transform = transform
        self.device = device

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images.

        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Returns the image and its corresponding name at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            tuple: A tuple containing the image name and the processed image.

        """
        img_path = self.img_paths[idx]
        img_name = img_path.stem
        image = Image.open(img_path)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image).to(self.device)

        return img_name, image


class CustomMultipleDataset(CustomDataset):
    def __init__(
        self,
        img_dir_1: Path,
        img_dir_2: Path,
        device: str,
        transform: transforms.Compose = None,
        num_images: int = None,
    ):
        """
        Initialize the CustomDataset class.

        Args:
            img_dir_1 (Path): Path to the first directory containing images.
            img_dir_2 (Path): Path to the second directory containing images.
            device (str): Device to be used for processing (e.g., 'cpu', 'cuda').
            transform (transforms.Compose, optional): Optional transformation to be applied to the images. Defaults to None.
            num_images (int, optional): Number of images to be loaded. If not provided, all images will be loaded. Defaults to None.
        """
        self.img_paths = self.__get_image_samples(img_dir_1, img_dir_2, num_images)
        self.transform = transform
        self.device = device

    def __get_image_samples(
        self, img_dir_1: Path, img_dir_2: Path, num_images: int
    ) -> list:
        """
        Randomly samples a specified number of images from two directories and returns a list of the sampled images.

        Args:
            img_dir_1 (Path): The path to the first directory containing images.
            img_dir_2 (Path): The path to the second directory containing images.
            num_images (int): The number of images to sample. If not provided, it will sample from all available images.

        Returns:
            list: A list of the sampled images.

        """
        images_1 = list(img_dir_1.glob("*"))
        images_2 = list(img_dir_2.glob("*"))
        total_images_1 = len(images_1)
        total_images_2 = len(images_2)

        if not num_images:
            num_images = total_images_1 + total_images_2

        half_num_images = num_images // 2

        num_images_1 = min(half_num_images, total_images_1)
        num_images_2 = min(half_num_images, total_images_2)

        if num_images_1 + num_images_2 < num_images:
            remaining_images = num_images - (num_images_1 + num_images_2)
            if total_images_1 - num_images_1 >= remaining_images:
                num_images_1 += remaining_images
            else:
                num_images_2 += remaining_images

        sampled_images_1 = random.sample(images_1, num_images_1)
        sampled_images_2 = random.sample(images_2, num_images_2)
        final_images = sampled_images_1 + sampled_images_2
        random.shuffle(final_images)

        return final_images
