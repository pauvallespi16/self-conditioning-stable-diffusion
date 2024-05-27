from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
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
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.stem
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image).to(self.device)

        return img_name, image
