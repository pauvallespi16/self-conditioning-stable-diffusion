from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset, CustomMultipleDataset
from utils import *
from wrapper import Process
from wrapper_sd import StableDiffusionWrapper
from wrapper_vae import VariationalAutoencoderWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = Path("images")
DOG_DATASET_PATH = DATASET_PATH / "dogs"

WRAPPERS_DICT = {"vae": VariationalAutoencoderWrapper, "sd": StableDiffusionWrapper}
SCORES_DICT = {"auroc": compute_auroc, "ap": compute_average_precision}


def generate(
    generate_dataset_path: Path,
    evaluate_dataset_path: Path,
    score_type: str = "auroc",
    output_activations_path: Path = None,
    output_layer_scores_path: Path = None,
) -> Tuple[
    Dict[str, Tuple[List[float], List[float]]], Dict[str, Dict[str, List[float]]]
]:
    """
    Generate activations for the VAE model.

    Args:
        generate_dataset_path (Path): Path to the dataset for generating activations.
        evaluate_dataset_path (Path): Path to the dataset for generating activations of the evaluation set.
        score_type (str, optional): The score to compute. Defaults to "auroc".
        output_activations_path (Path, optional): Path to save the activations. Defaults to None.
        output_layer_scores_path (Path, optional): Path to save the average precision scores. Defaults to None.

    Returns:
        agg_activations (Dict[str, Tuple[List[float], List[float]]]): Dictionary containing the aggregated activations of the layers.
        layer_scores (Dict[str, List[float]]): Dictionary containing the average precision scores of the activations.

    """
    torch.manual_seed(42)
    dataset = CustomMultipleDataset(
        generate_dataset_path,
        evaluate_dataset_path,
        device=DEVICE,
        transform=get_transforms(),
        num_images=5
    )
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
    agg_activations = compute_aggregated_activations(VAE_LAYER_NAMES, activations)
    save_pickle(agg_activations, output_activations_path)

    layer_score_function = SCORES_DICT[score_type]
    layer_scores = layer_score_function(VAE_LAYER_NAMES, activations)
    save_pickle(layer_scores, output_layer_scores_path)

    return agg_activations, layer_scores


def evaluate(
    dataset_path: Path,
    agg_activations: Union[Dict[str, Tuple[List[float], List[float]]], Path],
    layer_scores: Union[Dict[str, List[float]], Path],
    output_images_folder: Path = None,
):
    """
    Evaluate the VAE model using pre-generated activations.

    Args:
        dataset_path (Path): Path to the dataset.
        agg_activations (Union[Dict[str, Tuple[List[float], List[float]]], Path]): Dictionary containing the aggregated activations of the layers.
        layer_scores (Union[Dict[str, List[float]], Path]): Dictionary containing the scores of the activations.
        output_images_folder (Path, optional): Folder to save output images. Defaults to None.
    """
    agg_activations = load_pickle(agg_activations)
    layer_scores = load_pickle(layer_scores)

    dataset = CustomDataset(
        dataset_path, device=DEVICE, transform=get_transforms(), num_images=10
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

    vae.register_hooks(vae.vae, agg_activations, layer_scores)
    vae.run(dataloader)


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--architecture",
        type=str,
        default="vae",
        choices=["vae", "sd"],
        help="The architecture to use.",
    )
    parser.add_argument(
        "--generate_dataset_path",
        type=Path,
        default=DOG_DATASET_PATH,
        help="The path to the dataset for generating activations.",
    )
    parser.add_argument(
        "--evaluate_dataset_path",
        type=Path,
        default=DOG_DATASET_PATH,
        help="The path to the dataset for evaluating the model.",
    )
    parser.add_argument(
        "--process",
        type=str,
        default="generation",
        choices=["generation", "evaluation", "both"],
        help="The process to run.",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        default="auroc",
        choices=["auroc", "ap"],
        help="The score to compute. Either AUROC (auroc) or Average Precision (ap).",
    )
    parser.add_argument(
        "--aggregated_activations_path",
        type=Path,
        default=None,
        help="The path to load/save the activations.",
    )
    parser.add_argument(
        "--layer_scores_path",
        type=Path,
        default=None,
        help="The path to load/save the scores of the activations.",
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
    generate_dataset_path = args.generate_dataset_path
    evaluate_dataset_path = args.evaluate_dataset_path
    process = args.process
    score_type = args.score_type
    aggregated_activations_path = args.aggregated_activations_path
    layer_scores_path = args.layer_scores_path
    output_images_folder = args.output_images_folder

    if process == "both":
        agg_activations, layer_scores = generate(
            generate_dataset_path,
            evaluate_dataset_path,
            score_type=score_type,
            output_activations_path=aggregated_activations_path,
            output_layer_scores_path=layer_scores_path,
        )
        evaluate(
            evaluate_dataset_path,
            agg_activations=agg_activations,
            layer_scores=layer_scores,
            output_images_folder=output_images_folder,
        )

    elif process == "generation":
        generate(
            generate_dataset_path,
            evaluate_dataset_path,
            score_type=score_type,
            output_activations_path=aggregated_activations_path,
            output_layer_scores_path=layer_scores_path,
        )

    elif process == "evaluation":
        evaluate(
            evaluate_dataset_path,
            agg_activations=aggregated_activations_path,
            layer_scores=layer_scores_path,
            output_images_folder=output_images_folder,
        )
