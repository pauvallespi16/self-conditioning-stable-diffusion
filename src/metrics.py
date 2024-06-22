"""
This file contains functions adapted from the paper "Self-Conditioning Pre-Trained Language Models" by Apple Inc. 
The functions have been slightly modified for this project.
For more information, refer to the original repository: https://github.com/apple/ml-selfcond
"""

from functools import partial
from itertools import repeat
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def _single_response_ap(unit_response: Sequence[float], labels: Sequence[int]) -> float:
    return average_precision_score(y_true=labels, y_score=unit_response)


def compute_average_precision(
    responses: Mapping[str, Sequence[float]], labels: Sequence[int], cpus: int = None
) -> Dict[str, List[float]]:
    """
    Compute average precision between responses and labels
    Args:
        responses: A dict `{response_name: response_tensor}`, where `response_tensor` is [num_units, num_sentences]
        labels: Label for each sentence, has length `num_sentences`.
        cpus: Number of cpu's for multithreading.

    Returns:
        dict: {response_name, List[float] of length num_units}

    """
    aps = {}
    cpus = min(cpu_count() - 1, 8) if cpus is None else cpus
    pool = Pool(processes=cpus)
    sorted_layers = sorted(responses.keys())
    for layer in tqdm(
        sorted_layers,
        total=len(responses),
        desc=f"Computing Average Precision [{cpus} workers]",
    ):
        aps[layer] = pool.map(
            partial(_single_response_ap, labels=labels), responses[layer].T
        )
    pool.close()
    return aps


def _compute_auroc_chunk(responses: np.ndarray, labels: np.ndarray):
    # Compute and return AUROC scores for the chunk of data
    return roc_auc_score(
        labels[:, None].repeat(responses.shape[1], 1),
        responses,
        average=None,
    )


def compute_auroc(
    responses: Mapping[str, Sequence[float]],
    labels: Sequence[int],
    cpus: int = None,
    chunk_size: int = 10,
) -> Dict[str, List[float]]:
    """
    Compute the Area Under the Receiver Operating Characteristic (AUROC) for each layer in the responses.

    Args:
        responses (Mapping[str, Sequence[float]]): A mapping of layer names to response sequences.
        labels (Sequence[int]): The true labels for the responses.
        cpus (int, optional): The number of CPU processes to use for parallel computation. If not provided, it will use all available CPUs - 1 or a maximum of 8.
        chunk_size (int, optional): The size of each chunk to process in parallel.

    Returns:
        Dict[str, List[float]]: A dictionary mapping each layer name to a list of AUROC values.
    """
    aps = {}
    cpus = min(cpu_count() - 1, 8) if cpus is None else cpus
    pool = Pool(processes=cpus)
    sorted_layers = sorted(responses.keys())
    for layer in tqdm(
        sorted_layers, total=len(responses), desc=f"Computing AUROC [{cpus} workers]"
    ):
        responses_map = [
            responses[layer][:, start : (start + chunk_size)]
            for start in np.arange(0, responses[layer].shape[1], chunk_size)
        ]
        args = zip(responses_map, repeat(labels))
        ret = pool.starmap(_compute_auroc_chunk, args)
        aps[layer] = np.concatenate(ret, 0)

    return aps
