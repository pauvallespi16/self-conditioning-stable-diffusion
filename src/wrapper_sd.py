from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from wrapper import ModelWrapper, Process


class StableDiffusionWrapper(ModelWrapper):
    """
    Wrapper class for the Stable Diffusion model.

    Args:
        model_name (str): The name of the Stable Diffusion model.
        layers (List[str]): List of layer names to extract activations from.
        device (str): The device to run the model on (e.g., "cpu", "cuda").
        process (Process): The process to use for generating activations.
        sequence_length (int, optional): The desired sequence length for padding. Defaults to 50.
        aggregation_type (str, optional): The type of aggregation to apply to the activations. Defaults to "max".
        output_images_folder (Path, optional): The folder to save output images. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        layers: List[str],
        device: str,
        process: Process,
        sequence_length: int = 50,
        aggregation_type: str = "max",
        threshold: float = 0.8,
        output_images_folder: Path = None,
    ):
        super().__init__(
            model_name=model_name,
            layers=layers,
            device=device,
            process=process,
            output_images_folder=output_images_folder,
        )
        self.activations = {}
        self.sd = DiffusionPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16, variant="fp16"
        )
        self.threshold = threshold
        self.sd.enable_model_cpu_offload()
        self.sd.set_progress_bar_config(disable=True)
        self.text_encoder = self.sd.text_encoder
        self.tokenizer = self.sd.tokenizer
        self.text_encoder.eval()
        self.sequence_length = sequence_length
        self.aggregation_type = aggregation_type

    def generation_hook(self, *args):
        """
        Returns a hook function for the generation process.

        Args:
            *args: Additional arguments.

        Returns:
            Callable: The hook function.

        """

        def hook_fn(module, input, output):
            module_name = self.module_to_name[module]
            if module_name not in self.activations:
                self.activations[module_name] = []

            for i in range(output.shape[0]):
                out = (
                    output[i].detach().numpy()
                    if self.device == "cpu"
                    else output[i].detach().cpu().numpy()
                )
                self.activations[module_name].append(out)

        return hook_fn

    def evaluation_hook(self, *args):
        """
        Returns a hook function for the evaluation process.

        Args:
            *args: Additional arguments.

        Returns:
            Callable: The hook function.

        """
        layer_scores = args[0]

        def hook_fn(module, input, output):
            module_name = self.module_to_name[module]
            mask = layer_scores[module_name] > self.threshold
            self.metadata[module_name] = 100 * mask.sum() / mask.size
            output[:, :, mask] = 0
            return output

        return hook_fn

    def _aggregate_activations(
        self, activations_dict: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """
        Aggregates activations for each layer in the model.

        Args:
            activations_dict (Dict[str, List[float]]): Given activations.

        Returns:
            Dict[str, List[float]]: Dictionary containing the aggregated activations for each layer.
        """

        for layer in self.layers:
            activations_dict[layer] = np.array(activations_dict[layer])
            agg_fn = getattr(np, self.aggregation_type)
            # Shape -> 200 x 768
            # AP    -> 768 x 200
            # AUROC -> 200 x 768
            activations_dict[layer] = agg_fn(activations_dict[layer], axis=1)
        return activations_dict

    def _remove_too_long_data(
        self, data: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """
        Removes data that is too long.

        Args:
            data (Dict[str, List[float]]): The output of the tokenizer.

        Returns:
            Dict[str, List[float]]: Dictionary without the too long data.
        """

        remove_idx = []
        for idx, tokens in enumerate(data["input_ids"]):
            extra_tokens = 0
            if len(tokens) > self.sequence_length + extra_tokens:
                remove_idx.append(idx)

        remove_idx = sorted(remove_idx, reverse=True)

        for key in data.keys():
            for i in remove_idx:
                del data[key][i]

        return data

    def _pad_indexed_tokens(
        self, indexed_tokens: List[int], min_num_tokens: int
    ) -> List[int]:
        """
        Adds padding tokens to a list of token indices. For example:
        ```
        idx = [1, 2, 3] # and pad token is 100
        pad_idx = pad_indexed_tokens(idx, 5)
        > pad_idx: [1, 2, 3, 5, 5]
        ```

        Args:
            indexed_tokens: List of indexed tokens.
            min_num_tokens: Final number of tokens required, including padding.

        Returns:
            list: Indexed tokens padded.

        """
        assert min_num_tokens is not None
        assert min_num_tokens > 0

        # Get the padding token
        pad_token_id: int = self.tokenizer.pad_token_id

        # Actually pad sequence.
        num_effective_tokens = len(indexed_tokens)
        pad_tokens: int = max(min_num_tokens - num_effective_tokens, 0)
        return indexed_tokens + [pad_token_id] * pad_tokens

    def _preprocess_sequence(
        self, text: str, min_num_tokens: int = None
    ) -> Dict[str, List]:
        """
        Pre-processes a text sequence by applying a tokenizer and padding up till min_num_tokens tokens.
        The final sequence will have then (min_num_tokens) tokens.

        Example:
            ```
            idx, named_data = pre_process_sequence('My name is John', 10)
            # Generates the following tokens
            ['my', 'name', 'is', 'john', [PAD], [PAD], [PAD], [PAD], [PAD], [PAD]]
            # And returns their corresponding ids in the vocabulary associated with the tokenizer.
            # named_data['attention_mask'] is [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            ```
        Args:
            text: The text to be pre-processed
            min_num_tokens: If not None, the tokens sequence will be padded with padding_str up till min_num_tokens.

        Returns:
            output_tokenizer: output given by the tokenizer
        """
        indexed_tokens: List[int] = self.tokenizer.encode(text)

        num_effective_tokens = len(indexed_tokens)
        if min_num_tokens is not None:
            indexed_tokens = self._pad_indexed_tokens(indexed_tokens, min_num_tokens)

        assert len(indexed_tokens) >= num_effective_tokens
        attention_mask: List[int] = [1] * num_effective_tokens + [0] * (
            len(indexed_tokens) - num_effective_tokens
        )
        assert len(attention_mask) == len(indexed_tokens)

        output_tokenizer = {
            "input_ids": indexed_tokens,
            "attention_mask": attention_mask,
        }
        return output_tokenizer

    def preprocess_dataset(
        self, sentence_list: List[str], min_num_tokens: int = None
    ) -> Dict[str, List]:
        """
        Pre-proces a list of sentences.

        Args:
            sentence_list: List of sentences.
            min_num_tokens: Min number of tokens a sentence will have, if shorter it will be padded.

        Returns:
            input_model: dict of data that will be fed as kwargs to the model.
        """
        input_model: Dict[str, List] = defaultdict(list)
        for seq in sentence_list:
            if not isinstance(seq, str):
                continue

            input_model_seq = self._preprocess_sequence(
                text=seq,
                min_num_tokens=min_num_tokens,
            )

            for k, v in input_model_seq.items():
                input_model[k].append(v)

        for k in input_model.keys():
            input_model[k] = torch.tensor(input_model[k]).to(self.device)

        return input_model

    def generation(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        """
        Generates activations for each layer in the Stable Diffusion model.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            Dict[str, List[float]]: Dictionary containing the activations for each layer.
        """
        targets = []
        desc = GENERATION_STRING
        activations_dict = defaultdict(list)

        for sentences, labels in tqdm(dataloader, desc=desc):
            preprocessed_batch = self.preprocess_dataset(
                sentences, min_num_tokens=self.sequence_length
            )
            preprocessed_batch = self._remove_too_long_data(preprocessed_batch)
            self.text_encoder(**preprocessed_batch)

            targets.extend(labels)
            for module, activations in self.activations.items():
                activations_dict[module].extend(activations)

            self.activations.clear()

        targets = np.array(targets)
        activations_dict = self._aggregate_activations(activations_dict)
        return activations_dict, targets

    def inference(self, dataloader: DataLoader, save_metadata: bool = False):
        """
        Generates images using the Stable Diffusion model.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            save_metadata (bool, optional): Whether to save metadata. Defaults to False.
        """
        desc = EVALUATION_STRING

        for prompts, _ in tqdm(dataloader, desc=desc):
            sentences = list(prompts)
            output = self.sd(prompt=sentences, height=512, width=512).images
            save_images(self.output_images_folder, output, sentences)

        if save_metadata and self.process != Process.EVALUATION:
            save_json(
                f"metadata/{self.model_name.split('/')[-1]}_{self.threshold}.json",
                self.metadata,
            )
