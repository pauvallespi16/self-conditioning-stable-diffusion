from pathlib import Path

from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms

from utils import load_images


def compute_fid(
    real_images_path: Path, generated_images_path: Path, device: str = "cuda"
) -> float:
    """
    Compute the Frechet Inception Distance (FID) between real and generated images.

    Args:
        real_images_path (str): The path to the real images.
        generated_images_path (str): The path to the generated images.

    Returns:
        float: The FID score.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )

    real_images = load_images(real_images_path, transform)
    generated_images = load_images(generated_images_path, transform)

    fid = FrechetInceptionDistance(device=device)
    fid.update(real_images, is_real=True)
    fid.update(generated_images, is_real=False)

    return fid.compute().detach().cpu().numpy()
