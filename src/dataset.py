import os
from typing import Optional, List, Literal
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import (
    DataCollatorForTokenClassification,
    pad_without_fast_tokenizer_warning,
)
from dataclasses import dataclass, field
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer


class CachedPickleDatasetV2(Dataset):
    """
    A dataset that caches the data in a pickle file.

    Args:
        data_dir (`str`):
            The directory to the data.
        verbose (`bool`):
            Whether to print the verbose information.
        indices (`List[int]`, *optional*, defaults to `None`):
            The indices of the data to select.
    """

    def __init__(
        self,
        data_dir: str,
        verbose: bool = False,
        test_indices: Optional[List[int]] = None,
        include_gt: bool = True,
        get_single_sample: bool = True,
        remove_hard_samples: bool = False,
        remove_easy_samples: bool = False,
        device: str | int = "cpu",
    ):
        self.data_dir = data_dir
        self.verbose = verbose
        self.include_gt = include_gt
        self.get_single_sample = get_single_sample
        self.remove_hard_samples = remove_hard_samples
        self.device = device
        all_files = [f for f in os.listdir(data_dir) if f.endswith(".safetensors")]
        self.idx_to_file_name = {}
        self.skipped_num_tests = 0
        self.n_samples = None
        for fname in tqdm(all_files, total=len(all_files)):
            with safe_open(os.path.join(data_dir, fname), framework="pt", device=self.device) as f:
                _keys = list(f.keys())
                valid_idxes = []
                all_idxes = []
                hard_idxes = []
                easy_idxes = []
                for k in _keys:
                    if "estimations" in k:
                        valid_idxes.append(int(k.split(".")[0]))
                    if "input_ids" in k:
                        all_idxes.append(int(k.split(".")[0]))
                    if (remove_hard_samples or remove_easy_samples) and "corrects" in k:
                        tensor = f.get_tensor(k)
                        if tensor.sum() == 0 and remove_hard_samples:
                            hard_idxes.append(int(k.split(".")[0]))
                        elif tensor.sum() == tensor.shape[0] and remove_easy_samples:
                            easy_idxes.append(int(k.split(".")[0]))
                valid_idxes = list(set(valid_idxes) - set(hard_idxes) - set(easy_idxes))
                all_idxes = list(set(all_idxes))
                self.skipped_num_tests += len(all_idxes) - len(valid_idxes)
                if not len(valid_idxes):
                    continue
                for t in valid_idxes:
                    self.idx_to_file_name[t] = fname

                if self.n_samples is None:
                    estimations = f.get_slice(f"{valid_idxes[0]}.estimations")
                    self.n_samples = estimations.get_shape()[0]
                    if self.verbose:
                        print(f"n_samples: {self.n_samples}")
        assert self.n_samples is not None, "n_samples is not set"
        if self.verbose:
            print(f"Loaded {self.num_test_samples} test prompts")
            if remove_hard_samples:
                print(f"(Additionally removed {len(hard_idxes)} hard samples)")
            if remove_easy_samples:
                print(f"(Additionally removed {len(easy_idxes)} easy samples)")
            print(
                f"Original: {self.skipped_num_tests + self.num_test_samples} | "
                f"Skipped: {self.skipped_num_tests} "
            )

        if test_indices is not None:
            new_idx_to_file_name = {}
            num_not_found = 0
            for k in test_indices:
                if k in self.idx_to_file_name:
                    new_idx_to_file_name[k] = self.idx_to_file_name[k]
                else:
                    num_not_found += 1
                    # print(f"Warning: {k} not found in {self.idx_to_file_name}")
            if num_not_found > 0:
                print(f"Warning: {num_not_found} indices not found in the dataset")
            self.idx_to_file_name = new_idx_to_file_name

            if self.verbose:
                print(f"Selected {self.num_test_samples} test prompts")

        self.idxes = list(self.idx_to_file_name.keys())

    def __len__(self):
        return self.num_test_samples * (self.n_samples if self.get_single_sample else 1)

    @property
    def num_test_samples(self):
        return len(self.idx_to_file_name)

    def __str__(self) -> str:
        return f"DatasetV2 (num_input_ids={self.num_test_samples}, n_samples={self.n_samples}, device={self.device})"

    def __getitem__(self, idx):

        if self.get_single_sample:
            sample_idx = self.idxes[idx // self.n_samples]
            inner_idx = idx % self.n_samples
        else:
            sample_idx = self.idxes[idx]
            inner_idx = slice(0, self.n_samples)
        file_name = self.idx_to_file_name[sample_idx]
        file_path = os.path.join(self.data_dir, file_name)

        with safe_open(file_path, framework="pt", device=self.device) as f:
            latent_embeds = f.get_slice(f"{sample_idx}.latent_embeds")
            latent_embeds = latent_embeds[inner_idx]

            input_ids = f.get_tensor(f"{sample_idx}.input_ids")
            if input_ids.shape[0] == 1:
                input_ids = input_ids.squeeze(0)
            if input_ids.dim() == 1:
                if not self.get_single_sample:
                    input_ids = input_ids.unsqueeze(0).repeat(self.n_samples, 1)
            else:
                input_ids = input_ids[inner_idx]

            estimations = f.get_slice(f"{sample_idx}.estimations")
            estimations = estimations[inner_idx]
            result = {
                "latent_embeds": latent_embeds,
                "input_ids": input_ids,
                "estimations": estimations,
            }

            if self.include_gt:
                try:
                    corrects = f.get_slice(f"{sample_idx}.corrects")
                    corrects = corrects[inner_idx]
                    result["corrects"] = corrects
                except:
                    pass
            return result

    def select(self, indices):
        return CachedPickleDatasetV2(
            self.data_dir,
            verbose=self.verbose,
            test_indices=indices,
            get_single_sample=self.get_single_sample,
            include_gt=self.include_gt,
            remove_hard_samples=self.remove_hard_samples,
            device=self.device,
        )


@dataclass
class DataCollatorForLatentRM(DataCollatorForTokenClassification):
    latent_token_id: int = -1
    latent_end_id: int = -1
    generator_tokenizer: AutoTokenizer | None = None
    remove_pad_token: bool = True
    postfix: str = 6 * "<|latent|>" + "<|end-latent|>"
    encoded_postfix: List[int] = field(default_factory=lambda: 6 * [50259] + [50258])

    def torch_call(self, features):

        for feature in features:
            input_ids = feature["input_ids"]
            if self.generator_tokenizer is not None:
                inputs = self.generator_tokenizer.decode(input_ids, skip_special_tokens=True)

                if self.postfix not in inputs:
                    inputs = inputs + self.postfix
                splited = inputs.split("\n")
                question = splited[0]
                answer = "\n".join(splited[1:])
                input_ids = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )

            elif self.remove_pad_token:
                input_ids = input_ids[input_ids != self.tokenizer.pad_token_id]

            if self.latent_token_id not in input_ids:
                input_ids = torch.cat([input_ids, torch.tensor(self.encoded_postfix)])
            feature["input_ids"] = input_ids

        feature_names = ["input_ids"]
        no_labels_features = [
            {k: v for k, v in feature.items() if k in feature_names} for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["latent_embeds"] = [feature["latent_embeds"] for feature in features]

        labels = torch.full_like(batch["input_ids"], -100, dtype=batch["latent_embeds"][0].dtype)
        for i in range(len(batch["latent_embeds"])):

            latent_indices = batch["input_ids"][i] == self.latent_token_id
            estimations = features[i]["estimations"]
            num_latent_steps = latent_indices.long().sum()

        if "corrects" in features[i]:
            corrects = features[i]["corrects"]
            # find the largest (last) non -100 index
            last_non_100_idx = (estimations != -100).nonzero()[-1]
            estimations[last_non_100_idx] = corrects.float()

            labels[i, latent_indices] = estimations[:num_latent_steps]

        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorForContrastiveLatentRM(DataCollatorForLatentRM):

    def torch_call(self, features):
        all_input_ids = []
        for feature in features:
            input_ids = feature["input_ids"]
            if self.generator_tokenizer is not None:
                inputs = self.generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                modified_inputs = []
                for _in in inputs:
                    if self.postfix not in _in:
                        _in = _in + self.postfix

                    splited = _in.split("\n")
                    question = splited[0]
                    answer = "\n".join(splited[1:])
                    _input_ids = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                modified_inputs.append(_input_ids)  # List[List[int]]
            else:
                modified_inputs = [input_ids[i] for i in range(len(input_ids))]
            for i in range(len(modified_inputs)):
                if self.latent_token_id not in modified_inputs[i]:
                    modified_inputs[i] = torch.cat([modified_inputs[i], torch.tensor(self.encoded_postfix)])

            all_input_ids.extend(modified_inputs)

        n_samples = input_ids.shape[0]

        no_labels_features = [{"input_ids": v} for v in all_input_ids]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["latent_embeds"] = [feature["latent_embeds"] for feature in features]

        labels = torch.full_like(batch["input_ids"], -100, dtype=batch["latent_embeds"][0].dtype)
        for i in range(len(batch["latent_embeds"])):
            cur_input_ids = batch["input_ids"][i * n_samples : (i + 1) * n_samples]

            latent_indices = cur_input_ids[0] == self.latent_token_id
            estimations = features[i]["estimations"]
            num_latent_steps = latent_indices.long().sum()

            if "corrects" in features[i]:
                corrects = features[i]["corrects"]
                # find the largest (last) non -100 index
                for j in range(n_samples):
                    last_non_100_idx = (estimations[j] != -100).nonzero()[-1]
                    estimations[j, last_non_100_idx] = corrects[j].float()

            labels[i * n_samples : (i + 1) * n_samples, latent_indices] = estimations[:, :num_latent_steps]

        batch["labels"] = labels

        return batch
