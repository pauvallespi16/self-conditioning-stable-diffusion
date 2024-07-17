from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from transformers import pipeline

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CLIP:
    """
    A wrapper class for the CLIP model.

    Args:
        model_name (str): The name of the CLIP model.
        labels (List[str]): A list of labels for zero-shot classification.
        device (str): The device to run the model on (e.g., "cpu", "cuda").

    Attributes:
        clip (pipeline): The CLIP pipeline.
        labels (List[str]): A list of labels for zero-shot classification.
    """

    def __init__(self, model_name: str, labels: List[str], device: str = "cpu"):
        self.device = device
        self.labels = labels
        self.model_name = model_name
        self.clip = pipeline(
            model=self.model_name,
            task="zero-shot-image-classification",
            device=self.device,
        )

    def predict(self, input: Union[Image.Image, List[Image.Image]]) -> List[dict]:
        """
        Predicts the labels of the input images.

        Args:
            input (Union[Image.Image, List[Image.Image]]): The input image(s).

        Returns:
            List[dict]: A list of dictionaries containing the predicted labels.
        """
        return self.clip(input, candidate_labels=self.labels)

    def score(self, images_path: Path, metric: CLIPScore) -> List[float]:
        """
        Scores the images in the given path.

        Args:
            images_path (Path): The path to the images.

        Returns:
            List[float]: A list of scores for the images.
        """
        paths = list(images_path.glob("*"))

        images = [
            torch.from_numpy(np.array(Image.open(image))).permute(2, 0, 1)
            for image in paths
        ]
        prompts = [str(prompt).split(".")[0] for prompt in paths]

        return metric(images, prompts)


def evaluate_images(
    original_dataset: Path,
    output_dataset: Path,
    labels: List[str] = ["Pink Elephant", "Something Else"],
) -> Tuple[float, float, float, float]:
    """
    Evaluates the images in the original and output datasets.

    Args:
        original_dataset (Path): The path to the original dataset.
        output_dataset (Path): The path to the output dataset.
        labels (List[str], optional): A list of labels for zero-shot classification.
            The first element is the concept to evaluate, while the second element is the fallback concept.
            Defaults to ["Pink Elephant", "Something Else"].

    Returns:
        Tuple[List[float], float, float]: The percentage of a concept in the original and output datasets
            and the CLIP scores for the original and output datasets.
    """
    clip = CLIP(CLIP_MODEL_NAME, labels, device=DEVICE)
    # metric = CLIPScore(model_name_or_path=CLIP_MODEL_NAME)

    num_concepts = 2 * len(labels)
    original_images = list(original_dataset.glob("*"))
    output_images = list(output_dataset.glob("*"))
    pbar = tqdm(total=len(original_images), desc="Evaluating images...")

    concept_count = [0] * num_concepts
    for original, output in zip(original_images, output_images):
        original_image = Image.open(original)
        output_image = Image.open(output)

        original_predictions = clip.predict(original_image)
        output_predictions = clip.predict(output_image)

        for i in range(num_concepts, step=2):
            concept_count[i] += original_predictions[0]["label"] == labels[i // 2]
            concept_count[i + 1] += output_predictions[0]["label"] == labels[i // 2]

        pbar.update(1)

    for i in range(num_concepts, step=2):
        concept_count[i] = 100 * concept_count[i] / len(original_images)
        concept_count[i + 1] = 100 * concept_count[i + 1] / len(output_images)

    clip_score_original_dataset = 0  # clip.score(original_dataset, metric=metric)
    clip_score_output_dataset = 0  # clip.score(output_dataset, metric=metric)

    return (
        concept_count,
        clip_score_original_dataset,
        clip_score_output_dataset,
    )


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--original_dataset",
        type=Path,
        default=Path("data/original"),
        help="The path to the original dataset.",
    )
    parser.add_argument(
        "--output_dataset",
        type=Path,
        default=Path("data/output"),
        help="The path to the output dataset.",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)

    args = parser.parse_args()
    original_dataset = args.original_dataset
    output_dataset = args.output_dataset

    original_pink_elephants, output_pink_elephants = evaluate_images(
        original_dataset, output_dataset
    )
    print(
        f"Percentage of pink elephants in the original dataset: {original_pink_elephants:.3f}%"
    )
    print(
        f"Percentage of pink elephants in the output dataset: {output_pink_elephants:.3f}%"
    )
