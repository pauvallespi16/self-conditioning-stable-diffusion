from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
from utils import *
from wrapper import Process
from wrapper_sd import StableDiffusionWrapper
from wrapper_vae import VariationalAutoencoderWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_OUTPUT_IMAGES = False

DATASET_PATH = Path("images")

CAT_DATASET_PATH = DATASET_PATH / "cats"
DOG_DATASET_PATH = DATASET_PATH / "dogs"
COMBINED_DATASET_PATH = DATASET_PATH / "combined"

WRAPPERS_DICT = {"vae": VariationalAutoencoderWrapper, "sd": StableDiffusionWrapper}


def generate(
    dataset_path: Path,
    output_average_precision_path: Path = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Generate activations for the VAE model.

    Args:
        dataset_path (Path): Path to the dataset.
        output_average_precision_path (Path, optional): Path to save the average precision scores. Defaults to None.

    Returns:
        average_precision_scores (Dict[str, List[float]]): Dictionary containing the average precision scores of the activations.

    """
    torch.manual_seed(42)
    dataset = CustomDataset(dataset_path, device=DEVICE, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # TODO: Adapt to new models
    vae = VariationalAutoencoderWrapper(
        model_name=VAE_MODEL_NAME,
        layers=VAE_LAYER_NAMES,
        device=DEVICE,
        process=Process.GENERATION,
    )

    vae.register_hooks(vae.vae)
    activations = vae.run(dataloader)
    average_precision = compute_average_precision(VAE_LAYER_NAMES, activations)

    if output_average_precision_path:
        save_pickle(average_precision, output_average_precision_path)

    return average_precision


def evaluate(
    dataset_path: Path,
    average_precision: Union[Dict[str, List[float]], Path],
    output_images_folder: Path = None,
):
    """
    Evaluate the VAE model using pre-generated activations.

    Args:
        dataset_path (Path): Path to the dataset.
        average_precision (Union[Dict[str, List[float]], Path]): Dictionary containing the average precision scores of the activations.
        output_images_folder (Path, optional): Folder to save output images. Defaults to None.

    Returns:
        None
    """
    if isinstance(average_precision, Path):
        average_precision = load_pickle(average_precision)

    dataset = CustomDataset(
        dataset_path, device=DEVICE, transform=get_transforms(), num_images=1
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # TODO: Adapt to new models
    vae = VariationalAutoencoderWrapper(
        model_name=VAE_MODEL_NAME,
        layers=VAE_LAYER_NAMES,
        device=DEVICE,
        process=Process.EVALUATION,
        output_images_folder=output_images_folder,
    )

    print(average_precision)
    # TODO: Adapt to use ap_scores instead
    # agg_activations = {
    #     module: np.median(list(acts.values()), axis=0)
    #     for module, acts in tqdm(activations.items(), desc="Aggregating activations")
    # }
    # units = slice(0, 14)

    # vae.register_hooks(vae.vae, agg_activations)
    # vae.run(dataloader)


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--architecture",
        type=str,
        default="vae",
        choices=["vae", "sd"],
        help="The architecture to use.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=COMBINED_DATASET_PATH,
        help="The path to the dataset.",
    )
    parser.add_argument(
        "--process",
        type=str,
        default="generation",
        choices=["generation", "evaluation", "both"],
        help="The process to run.",
    )
    parser.add_argument(
        "--average_precision_path",
        type=Path,
        default=None,
        help="The path to load/save the average precision scores of the activations.",
    )
    parser.add_argument(
        "--output_images_folder",
        type=Path,
        default=None,
        help="The folder to save output images.",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)

    args = parser.parse_args()
    wrapper = WRAPPERS_DICT[args.architecture]
    dataset_path = args.dataset_path
    process = args.process
    average_precision_path = args.average_precision_path
    output_images_folder = args.output_images_folder

    if process == "both":
        average_precision = generate(
            dataset_path,
            output_average_precision_path=average_precision_path,
        )
        evaluate(
            dataset_path,
            average_precision=average_precision,
            output_images_folder=output_images_folder,
        )

    elif process == "generation":
        generate(
            dataset_path,
            output_average_precision_path=average_precision_path,
        )

    elif process == "evaluation":
        evaluate(
            dataset_path,
            average_precision_scores=average_precision_path,
            output_images_folder=output_images_folder,
        )
