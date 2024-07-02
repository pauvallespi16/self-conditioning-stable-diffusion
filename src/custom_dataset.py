import random
from pathlib import Path

from torch.utils.data import Dataset

random.seed(42)


class SentenceDataset(Dataset):
    """
    A custom dataset class for handling sentences.

    Args:
        sentences_path (Path): The path to the file containing the sentences.

    Attributes:
        sentences (list): A list of sentences.
        labels (list): A list of labels.
    """

    def __init__(self, sentences_path: Path):
        super().__init__()
        with open(sentences_path, "r") as f:
            self.sentences = f.read().split("\n")
            random.shuffle(self.sentences)
        self.labels = [0] * len(self.sentences)  # 0 for eval

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class MultipleSentenceDataset(SentenceDataset):
    """
    A custom dataset class for handling multiple sentences.

    Args:
        positive_dataset_path (Path): The path to the positive dataset file.
        negative_dataset_path (Path): The path to the negative dataset file.

    Attributes:
        sentences (list): A list of sentences from the combined dataset.
        labels (list): A list of labels corresponding to the sentences.
    """

    def __init__(self, positive_dataset_path: Path, negative_dataset_path: Path):
        with open(positive_dataset_path, "r") as f:
            positive_sentences = f.read().split("\n")
            positive_labels = [1] * len(positive_sentences)
        with open(negative_dataset_path, "r") as f:
            negative_sentences = f.read().split("\n")
            negative_labels = [0] * len(negative_sentences)

        combined = list(zip(positive_sentences, positive_labels)) + list(
            zip(negative_sentences, negative_labels)
        )
        random.shuffle(combined)

        self.sentences, self.labels = zip(*combined)
        self.sentences = list(self.sentences)
        self.labels = list(self.labels)  # 1 for gen, 0 for eval
