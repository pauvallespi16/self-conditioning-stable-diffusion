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

    def score(self, images_path: Path, batch_size: int = 8) -> List[float]:
        """
        Scores the images in the given path.

        Args:
            images_path (Path): The path to the images.

        Returns:
            List[float]: A list of scores for the images.
        """
        metric = CLIPScore(
            model_name_or_path=CLIP_MODEL_NAME,
            compute_on_cpu=True,
            compute_with_cache=True,
        )
        metric.to(DEVICE)
        paths = list(images_path.glob("*"))

        for i in range(0, len(paths), batch_size):
            images = torch.stack(
                [
                    torch.from_numpy(np.array(Image.open(image)))
                    .permute(2, 0, 1)
                    .to(DEVICE)
                    for image in paths[i : min(i + batch_size, len(paths))]
                ]
            )
            prompts = [f"An image of a {self.labels[0]}"] * len(images)
            metric.update(images, prompts)
            torch.cuda.empty_cache()

        return metric.compute().detach().cpu().numpy()


def evaluate_images(
    original_dataset: Path,
    output_dataset: Path,
    labels: List[str],
) -> Tuple[List[float], float, float]:
    """
    Evaluates the images in the original and output datasets.

    Args:
        original_dataset (Path): The path to the original dataset.
        output_dataset (Path): The path to the output dataset.
        labels (List[str], optional): A list of labels for zero-shot classification.
            The first element is the concept to evaluate, while the second element is the fallback concept.

    Returns:
        Tuple[List[float], float, float]: The percentage of a concept in the original and output datasets
            and the CLIP scores for the original and output datasets.
    """
    clip = CLIP(CLIP_MODEL_NAME, labels, device=DEVICE)

    num_concepts = 2 * len(labels)
    original_images = list(original_dataset.glob("*"))
    output_images = list(output_dataset.glob("*"))
    pbar = tqdm(total=len(original_images), desc="Evaluating images...")

    concept_count = [0] * num_concepts
    for original, output in zip(original_images, output_images):
        original_image = Image.open(original)
        output_image = Image.open(output)

        try:
            original_predictions = clip.predict(original_image)
            output_predictions = clip.predict(output_image)

            for i in range(0, num_concepts, 2):
                concept_count[i] += original_predictions[0]["label"] == labels[i // 2]
                concept_count[i + 1] += output_predictions[0]["label"] == labels[i // 2]

        except Exception as e:
            print(f"ERROR: Processing {original} and {output} with exception: {e}")

        pbar.update(1)

    for i in range(0, num_concepts, 2):
        concept_count[i] = 100 * concept_count[i] / len(original_images)
        concept_count[i + 1] = 100 * concept_count[i + 1] / len(output_images)

    clip_score_original_dataset = clip.score(original_dataset)
    clip_score_output_dataset = clip.score(output_dataset)

    return (
        concept_count,
        clip_score_original_dataset,
        clip_score_output_dataset,
    )
