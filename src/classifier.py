import random
from pathlib import Path
from typing import List, Tuple

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

random.seed(42)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_CLASSES = ["dog", "human"]

DOG_IMAGES_PATH = Path("images/dogs")
OTHER_IMAGES_PATH = Path("images/humans")


class CLIPClassifier:
    def __init__(
        self,
        model_name: str,
        images_path: List[Path],
        target_classes: List[str],
        device: str,
        save_plot_path: Path = None,
    ):
        """
        Initialize the Classifier object.

        Args:
            model_name (str): The name of the CLIP model to load.
            images_path (List[Path]): A list of paths to the images to classify.
            target_classes (List[str]): A list of target classes for classification.
            device (str): The device to use for inference (e.g., 'cpu', 'cuda').
            save_plot_path (Path, optional): The path to save the classification plot. Defaults to None.
        """
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.target_classes = target_classes
        self.device = device
        self.save_plot_path = save_plot_path
        self.images, self.image_embeddings = [], []
        self.__load_images(images_path)
        assert (
            len(self.images) == len(self.image_embeddings) and len(self.images) > 0
        ), "No images loaded."

    def __load_images(self, images_path: str):
        """
        Load and preprocess the images.

        Args:
            images_path (str): The path to the images.
        """
        for image in images_path:
            image = Image.open(image).convert("RGB")
            self.images.append(image)
            self.image_embeddings.append(self.preprocess(image))

    def encode_images(self) -> torch.Tensor:
        """
        Encode the images using the CLIP model.

        Returns:
            torch.Tensor: The encoded image features.
        """
        with torch.no_grad():
            image_input = torch.tensor(np.stack(self.image_embeddings)).to(self.device)
            image_features = self.model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def encode_texts(self) -> torch.Tensor:
        """
        Encode the target classes as text using the CLIP model.

        Returns:
            torch.Tensor: The encoded text features.
        """
        with torch.no_grad():
            text_descriptions = [
                f"This is a photo of a {label}" for label in self.target_classes
            ]
            text_tokens = clip.tokenize(text_descriptions).to(self.device)
            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the classification process.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the top probabilities and labels for each image.
        """
        image_features = self.encode_images()
        text_features = self.encode_texts()

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(len(self.target_classes), dim=-1)

        if self.save_plot_path:
            self.plot_results(self.images[:8], top_probs, top_labels)

        return top_probs, top_labels

    def plot_results(
        self, images: List[Image.Image], top_probs: np.ndarray, top_labels: np.ndarray
    ):
        """
        Plot the classification results.

        Args:
            images (List): A list of images to plot.
            top_probs (np.ndarray): The top probabilities for each image.
            top_labels (np.ndarray): The top labels for each image.
        """
        plt.figure(figsize=(16, 16))

        for i, image in enumerate(images):
            plt.subplot(8, 2, 2 * i + 1)
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(8, 2, 2 * i + 2)
            y = np.arange(top_probs.shape[-1])
            plt.grid()

            sorted_indices = top_labels[i].numpy().argsort()
            sorted_labels = top_labels[i].numpy()[sorted_indices]
            sorted_probs = top_probs[i].numpy()[sorted_indices]
            plt.barh(y, sorted_probs)
            plt.gca().invert_yaxis()
            plt.gca().set_axisbelow(True)
            plt.yticks(y, [self.target_classes[index] for index in sorted_labels])
            plt.xlabel("Probability")

        plt.subplots_adjust(wspace=0.5)
        plt.savefig(self.save_plot_path)


if __name__ == "__main__":
    dog_images_paths = list(DOG_IMAGES_PATH.glob("*"))
    other_images_paths = list(OTHER_IMAGES_PATH.glob("*"))
    reconstructed_images_path = list(Path("images/output").glob("*"))

    images_path = dog_images_paths + other_images_paths + reconstructed_images_path
    random.shuffle(images_path)

    images_path = images_path[:8]

    clip_classifier = CLIPClassifier(
        MODEL_NAME, images_path, TARGET_CLASSES, DEVICE, Path("plots/clip_results.png")
    )
    clip_classifier.run()
