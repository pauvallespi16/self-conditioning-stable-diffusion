import random
from pathlib import Path
from typing import List

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_CLASSES = ["dog", "human"]

DOG_IMAGES_PATH = Path("images/dogs")
OTHER_IMAGES_PATH = Path("images/humans")


def plot_results(images: List, top_probs: np.ndarray, top_labels: np.ndarray):
    plt.figure(figsize=(16, 16))

    for i, image in enumerate(images):
        plt.subplot(8, 2, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(8, 2, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [TARGET_CLASSES[index] for index in top_labels[i].numpy()])
        plt.xlabel("Probability")

    plt.subplots_adjust(wspace=0.5)
    plt.savefig("plots/clip_classifier.png")


def encode_images(image_embeddings: List[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        image_input = torch.tensor(np.stack(image_embeddings)).to(DEVICE)
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


def encode_texts() -> torch.Tensor:
    with torch.no_grad():
        text_descriptions = [
            f"This is a photo of a {label}" for label in TARGET_CLASSES
        ]
        text_tokens = clip.tokenize(text_descriptions).to(DEVICE)
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


if __name__ == "__main__":
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    images = []
    image_embeddings = []

    dog_images_paths = list(DOG_IMAGES_PATH.glob("*"))
    other_images_paths = list(OTHER_IMAGES_PATH.glob("*"))
    reconstructed_images_path = list(Path("images/output").glob("*"))
    images_path = dog_images_paths + other_images_paths + reconstructed_images_path
    random.shuffle(images_path)

    for image in images_path:
        image = Image.open(image).convert("RGB")
        images.append(image)
        image_embeddings.append(preprocess(image))

    image_features = encode_images(image_embeddings)
    text_features = encode_texts()

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(2, dim=-1)
    plot_results(images[520:528], top_probs, top_labels)
