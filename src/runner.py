from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader

from custom_dataset import MultipleSentenceDataset, SentenceDataset
from metrics import compute_auroc, compute_average_precision
from utils import *
from wrapper import Process
from wrapper_sd import StableDiffusionWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = Path("sentences")
SCORES_DICT = {"auroc": compute_auroc, "ap": compute_average_precision}


def generate(
    model_version: str,
    positive_dataset_path: Path,
    negative_dataset_path: Path,
    score_type: str = "auroc",
    aggregation_type: str = "max",
    output_activations_path: Path = None,
    output_layer_scores_path: Path = None,
) -> Tuple[
    Dict[str, Tuple[List[float], List[float]]],
    Dict[str, Dict[str, List[float]]],
]:
    torch.manual_seed(42)

    dataset = MultipleSentenceDataset(
        positive_dataset_path=positive_dataset_path,
        negative_dataset_path=negative_dataset_path,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sd = StableDiffusionWrapper(
        model_name=SD_VERSION_TO_MODEL[model_version],
        layers=SD_LAYERS,
        device=DEVICE,
        aggregation_type=aggregation_type,
        process=Process.GENERATION,
    )

    sd.register_hooks(sd.text_encoder)
    activations, labels = sd.generation(dataloader)
    save_pickle(activations, output_activations_path)

    layer_score_function = SCORES_DICT[score_type]
    layer_scores = layer_score_function(activations, labels)
    save_pickle(layer_scores, output_layer_scores_path)

    return activations, layer_scores


def evaluate(
    model_version: str,
    dataset_path: Path,
    activations: Union[Dict[str, List[float]], Path],
    layer_scores: Union[Dict[str, List[float]], Path],
    threshold: float = 0.75,
    output_images_folder: Path = None,
):
    activations = load_pickle(activations)
    layer_scores = load_pickle(layer_scores)

    dataset = SentenceDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sd = StableDiffusionWrapper(
        model_name=SD_VERSION_TO_MODEL[model_version],
        layers=SD_LAYERS,
        device=DEVICE,
        process=Process.EVALUATION,
        output_images_folder=output_images_folder,
    )

    sd.register_hooks(sd.text_encoder, layer_scores, threshold)
    sd.inference(dataloader)


def infer(
    model_version: str,
    dataset_path: Path,
    output_images_folder: Path = None,
):
    dataset = SentenceDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sd = StableDiffusionWrapper(
        model_name=SD_VERSION_TO_MODEL[model_version],
        layers=SD_LAYERS,
        device=DEVICE,
        process=Process.EVALUATION,
        output_images_folder=output_images_folder,
    )

    sd.inference(dataloader)


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--model_version",
        type=str,
        default="1.1",
        choices=["1.1", "1.5", "2.0", "xl-1.0"],
    )
    parser.add_argument(
        "--process",
        type=str,
        default="generation",
        choices=["generation", "evaluation", "generation_evaluation", "inference"],
        help="The process to run.",
    )
    parser.add_argument(
        "--positive_dataset_path",
        type=Path,
        default=DATASET_PATH / "positive_sentences.txt",
        help="The path to the dataset with the concept.",
    )
    parser.add_argument(
        "--negative_dataset_path",
        type=Path,
        default=DATASET_PATH / "negative_sentences.txt",
        help="The path to the dataset without the concept.",
    )
    parser.add_argument(
        "--inference_dataset_path",
        type=Path,
        default=DATASET_PATH / "positive_sentences.txt",
        help="The path to the dataset in which to run inference.",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        default="auroc",
        choices=["auroc", "ap"],
        help="The score to compute. Either AUROC (auroc) or Average Precision (ap).",
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="max",
        choices=["max", "mean", "median"],
        help="The aggregation type for the activations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="The threshold for the layer scores.",
    )
    parser.add_argument(
        "--activations_path",
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
    model_version = args.model_version
    positive_dataset_path = args.positive_dataset_path
    negative_dataset_path = args.negative_dataset_path
    inference_dataset_path = args.inference_dataset_path
    process = args.process
    score_type = args.score_type
    aggregation_type = args.aggregation_type
    threshold = args.threshold
    activations_path = args.activations_path
    layer_scores_path = args.layer_scores_path
    output_images_folder = args.output_images_folder

    if process == "generation_evaluation":
        activations, layer_scores = generate(
            model_version,
            positive_dataset_path,
            negative_dataset_path,
            score_type=score_type,
            aggregation_type=aggregation_type,
            output_activations_path=activations_path,
            output_layer_scores_path=layer_scores_path,
        )
        evaluate(
            model_version,
            inference_dataset_path,
            activations=activations,
            layer_scores=layer_scores,
            threshold=threshold,
            output_images_folder=output_images_folder,
        )

    elif process == "generation":
        generate(
            model_version,
            positive_dataset_path,
            negative_dataset_path,
            score_type=score_type,
            aggregation_type=aggregation_type,
            output_activations_path=activations_path,
            output_layer_scores_path=layer_scores_path,
        )

    elif process == "evaluation":
        evaluate(
            model_version,
            inference_dataset_path,
            activations=activations_path,
            layer_scores=layer_scores_path,
            threshold=threshold,
            output_images_folder=output_images_folder,
        )

    else:
        infer(
            model_version,
            inference_dataset_path,
            output_images_folder=output_images_folder,
        )
