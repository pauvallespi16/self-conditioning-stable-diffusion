from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    A custom dataset class for loading and processing images.

    Args:
        img_dir (Path): The directory containing the image files.
        device (str): The device to use for image processing.
        transform (callable, optional): A function/transform to apply to the images. Default is None.
        num_images (int, optional): The number of images to load. Default is 100. If -1 is pased, all images are used.

    Attributes:
        img_paths (list): A list of image file paths.
        transform (callable): The transform function to apply to the images.
        device (str): The device used for image processing.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image and its corresponding name at the given index.

    """

    def __init__(
        self, img_dir: Path, device: str, transform=None, num_images: int = 100
    ):
        images = list(img_dir.glob("*.jpg"))
        if num_images == -1:
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

        if self.transform:
            image = self.transform(image).to(self.device)

        return img_name, image
