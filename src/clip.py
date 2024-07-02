from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

from PIL import Image
from tqdm import tqdm
from transformers import pipeline

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


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
        self.clip = pipeline(
            model=model_name, task="zero-shot-image-classification", device=device
        )
        self.labels = labels

    def predict(self, input: Union[Image.Image, List[Image.Image]]) -> List[dict]:
        """
        Predicts the labels of the input images.

        Args:
            input (Union[Image.Image, List[Image.Image]]): The input image(s).

        Returns:
            List[dict]: A list of dictionaries containing the predicted labels.
        """
        return self.clip(input, candidate_labels=self.labels)


def evaluate_images(
    original_dataset: Path,
    output_dataset: Path,
    labels: List[str] = ["pink elephant", "something else"],
):
    """
    Evaluates the images in the original and output datasets.

    Args:
        original_dataset (Path): The path to the original dataset.
        output_dataset (Path): The path to the output dataset.
        labels (List[str], optional): A list of labels for zero-shot classification. Defaults to ["pink elephant", "something else"].

    Returns:
        Tuple[float, float]: The percentage of a concept in the original and output datasets.
    """
    clip = CLIP(CLIP_MODEL_NAME, labels)

    original_dataset = list(original_dataset.glob("*"))
    output_dataset = list(output_dataset.glob("*"))
    pbar = tqdm(total=len(original_dataset), desc="Evaluating images...")

    pink_elephants_count = [0, 0]
    for original, output in zip(original_dataset, output_dataset):
        original_image = Image.open(original)
        output_image = Image.open(output)

        original_predictions = clip.predict(original_image)
        output_predictions = clip.predict(output_image)

        pink_elephants_count[0] += original_predictions[0]["label"] == "pink elephant"
        pink_elephants_count[1] += output_predictions[0]["label"] == "pink elephant"
        pbar.update(1)

    pink_elephants_count[0] = 100 * pink_elephants_count[0] / len(original_dataset)
    pink_elephants_count[1] = 100 * pink_elephants_count[1] / len(output_dataset)
    return pink_elephants_count[0], pink_elephants_count[1]


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
