# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data Annotation Script for Latent TTS Models

This script generates annotated data for training and evaluation of latent TTS models.
It supports multiple model types (COCONUT, CODI, COLAR) and performs data generation
with latent thought estimation and answer validation.

Main Components:
- DataGenerator: Main class for generating annotated data
- Collator: Custom data collator for batching
- Answer extraction functions for different model types

Input Parameters for DataGenerator.__init__():

Core Parameters:
- use_wandb (bool, default=True): Whether to use Weights & Biases for logging
- seed (int, default=0): Random seed for reproducibility
- idx_file (str, default=""): Path to file containing indices to process/exclude
- excude_idxes (bool, default=True): Whether to exclude indices from idx_file (True) or only process them (False)
- monitor_interval (int, default=1): Interval for tqdm monitor updates
- dataset_piece (float, default=1.0): Fraction of dataset to use (0.0-1.0)
- dataset_indice (int, default=0): Which piece of the dataset to use when dataset_piece < 1.0
- datasets_progress_bar (bool, default=False): Whether to show progress bars for dataset operations
- model_type (Literal["coconut", "codi", "colar"], default="coconut"): Type of model to use
- remove_all_correct (bool, default=True): Whether to remove samples where all rollouts are correct
- remove_all_false (bool, default=True): Whether to remove samples where all rollouts are incorrect
- save_answers (bool, default=False): Whether to save answer outputs

Data Parameters:
- data_path (str, default="data/gsm_train.json"): Path to the input dataset
- name (str, default="debug"): Name for the output directory and wandb run
- save_path (str, default="latent-data"): Base directory for saving results

Generation Parameters:
- n_samples (int, default=8): Number of samples to generate per input
- n_samples_per_step (int, default=32): Number of samples for estimation steps
- batch_size (int, default=1024): Total batch size for processing
- max_new_tokens (int, default=64): Maximum number of new tokens to generate
- latent_length (int, default=6): Number of latent tokens to use

Saving Parameters:
- save_freq (int, default=-1): Frequency of saving results (-1 means save on every batch)
- save_answers (bool, default=False): Whether to save answer outputs

Debug Parameters:
- debug (bool, default=False): Whether to run in debug mode (uses smaller dataset)
- batch_progress_bar (bool, default=False): Whether to show progress bar for batch processing

Wandb Parameters:
- wandb_project (str, default="annotation-latent-data"): Weights & Biases project name

Generation Configuration (pass via with "generation_" prefix):
- generation_latent_do_sample (bool): Whether to sample latent tokens
- generation_latent_do_sample_by (str): Method for latent sampling ("dropout" or "noise")
- generation_dropout_p (float): Dropout probability for sampling
- generation_noise_std (float): Standard deviation for noise sampling

Usage:
    python annotate_data.py --model_type coconut --n_samples 8 --batch_size 1024
    python annotate_data.py --model_type colar --data_path data/test.json --debug True
"""

import os
import math
import gc
import warnings
from typing import Literal
from collections import Counter


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Third-party imports
import rich
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from accelerate import Accelerator
import datasets

import wandb

from .models.perturbation import find_and_replace_target_modules, MCLinear
from .paths import MODELS
from .utils import set_seed
from .generation_mixin import (
    LatentGenerationMixin,
    LatentGenerationConfig,
)


class Collator:
    def __init__(self, tokenizer):
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.padding = "longest"

    def __call__(self, features):
        feature_names = ["input_ids", "idx", "attention_mask"]

        no_labels_features = [
            {k: v for k, v in feature.items() if k in feature_names} for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["unpad_input_ids"] = [features[i]["input_ids"] for i in range(len(features))]
        if "answer" in features[0]:
            batch["answer"] = [features[i]["answer"] for i in range(len(features))]
        return batch


class DataGenerator:
    debug = False
    batch_progress_bar = False
    save_path = "latent-data"
    name = "debug"
    data_path = "data/gsm_train.json"
    n_samples_per_step = 32
    n_samples = 8
    batch_size = 1024
    save_freq = -1  # save on every batch
    save_answers = False

    wandb_run = None
    wandb_project: str = "annotation-latent-data"

    def __init__(
        self,
        use_wandb: bool = True,
        seed=0,
        idx_file="",
        excude_idxes: bool = True,
        monitor_interval=1,
        dataset_piece: float = 1.0,
        dataset_indice: int = 0,
        datasets_progress_bar: bool = False,
        model_type: Literal["coconut", "codi", "colar"] = "coconut",
        remove_all_correct: bool = True,
        remove_all_false: bool = True,
        save_answers: bool = False,
        **kwargs,
    ):
        self.model_type = model_type
        self.remove_all_correct = remove_all_correct
        self.remove_all_false = remove_all_false
        self.model_class = MODELS[model_type]["class"]
        self.model_id = MODELS[model_type]["id"]
        self.answer_extractor = MODELS[model_type]["answer_extractor"]
        if not datasets_progress_bar:
            datasets.disable_progress_bars()

        generation_kwargs = {
            "latent_do_sample_by": "dropout",
            "max_new_tokens": 64,
            "latent_length": 6,
        }
        for k, v in kwargs.items():
            if k.startswith("generation_"):
                generation_kwargs[k.replace("generation_", "")] = v
            else:
                setattr(self, k, v)

        # set tqdm monitor interval
        tqdm.monitor_interval = monitor_interval
        # set accelerator
        self.accelerator = Accelerator()

        if self.accelerator.is_main_process:
            if self.debug:
                rich.print("[bold red]DEBUG[/bold red]")
            if self.n_samples_per_step < 1 and self.remove_all_false:
                rich.print(
                    f"[bold red]WARNING: setting remove_all_false to False when n_samples_per_step == {self.n_samples_per_step}[/bold red]"
                )
                self.remove_all_false = False
            if use_wandb:
                rich.print("[bold green]Using wandb[/bold green]")
            config_dict = dict(
                seed=seed,
                idx_file=idx_file,
                dataset_piece=dataset_piece,
                dataset_indice=dataset_indice,
                n_samples_per_step=self.n_samples_per_step,
                n_samples=self.n_samples,
                model_type=self.model_type,
                remove_all_correct=self.remove_all_correct,
                remove_all_false=self.remove_all_false,
                save_answers=self.save_answers,
                data=self.data_path,
            )
            rich.print(config_dict)

            if len(generation_kwargs) > 0:
                rich.print(f"generation_kwargs: {generation_kwargs}")

        self.set_dirs()
        set_seed(seed)

        self.set_tokenizer_model(generation_kwargs)
        if self.accelerator.is_main_process and use_wandb:
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.name,
                config=config_dict
                | dict(
                    generation_config=self.generation_config.to_dict(),
                    estimation_generation_config=self.estimation_generation_config.to_dict(),
                ),
            )

        if self.accelerator.is_main_process:
            dataset = [
                self.prepare_dataset(
                    dataset_piece=dataset_piece,
                    dataset_indice=dataset_indice,
                    idx_file=idx_file,
                    excude_idxes=excude_idxes,
                )
            ]
        else:
            dataset = [None]

        self.accelerator.wait_for_everyone()
        dataset = self.accelerator.gather_for_metrics(dataset, use_gather_object=True)
        self.dataset = dataset[0]

        self.collator = Collator(self.tokenizer)

        self.answer_prefix_ids = torch.tensor(self.tokenizer.encode("### "), device=self.model.device)

        self.main_batch_size = self.batch_size // self.n_samples
        if self.debug:
            self.main_batch_size //= 16
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.main_batch_size, collate_fn=self.collator, shuffle=False
        )
        self.dataloader = self.accelerator.prepare(self.dataloader)

        self.accelerator.wait_for_everyone()

    def run(self):
        if self.accelerator.is_main_process:
            pbar = tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                colour="blue",
            )
        else:
            pbar = enumerate(self.dataloader)

        self.model.eval()
        self.results = {}
        self.num_saves = 0

        coverage = 0
        all_correct_count = 0
        all_false_count = 0
        voting_accuracy = 0
        total_num_samples = 0

        for batch_i, batch in pbar:
            model_inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            cur_batch_size, seq_len = model_inputs["input_ids"].shape
            if self.num_latents:
                assert (
                    seq_len > self.num_latents
                ), f"input_ids sequence length {seq_len} is less than num_latents {self.num_latents}"
            output = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                num_return_sequences=self.n_samples,
                return_dict_in_generate=True,
                concat_inputs=True,
            )
            input_ids_list = []
            for i in range(cur_batch_size):
                input_ids_list.append(torch.tensor(batch["unpad_input_ids"][i], device=self.model.device))

            output_ids = output.sequences.view(cur_batch_size, self.n_samples, -1)
            checked_outputs = self.check_output(
                output.sequences[:, seq_len:],
                batch["answer"],
                self.n_samples,
                return_voting_result=True,
                return_answer_output=self.save_answers,
            )  # (cur_B, n_samples)
            corrects = checked_outputs["corrects"]
            voting_result = checked_outputs["voting_result"]
            answer_output = checked_outputs.get("answer_output", None)
            total_num_samples += cur_batch_size

            coverage += corrects.any(dim=-1).sum().item()
            voting_accuracy += voting_result.sum().item()
            # all correct: a sample is correct if all rollouts are correct
            all_correct = corrects.all(dim=-1)
            all_correct_count += all_correct.sum().item()
            # all false: a sample is false if all rollouts are false
            all_false = (~corrects).all(dim=-1)
            all_false_count += all_false.sum().item()

            idxes_to_estimate = []
            input_ids_to_estimate = []
            answers_to_estimate = []
            thoughts_to_estimate = []
            output_ids_to_save = []
            for i in range(cur_batch_size):
                if self.remove_all_correct and all_correct[i].item():
                    continue
                if self.remove_all_false and all_false[i].item():
                    continue
                idxes_to_estimate.append(i)
                input_ids_to_estimate.append(input_ids_list[i])
                answers_to_estimate.append(batch["answer"][i])
                thoughts_to_estimate.append(
                    output.latent_thoughts[i * self.n_samples : (i + 1) * self.n_samples]
                )
                _out_ids = output_ids[i]  # (n_samples, L)
                # remove paddings
                """
                    e.g. -100 -100 x  x   x  -100
                            -100 -100 x  x -100 -100

                    the result should be
                    x  x   x
                    x  x
                    """
                # first delete the left paddings
                if (_out_ids[:, 0] == self.tokenizer.pad_token_id).all():
                    non_pad_mask = _out_ids != self.tokenizer.pad_token_id
                    first_non_pad = non_pad_mask.long().argmax(dim=-1).min().item()  # first non-padding index
                    _out_ids = _out_ids[:, first_non_pad:]
                # then delete the right paddings
                if (_out_ids[:, -1] == self.tokenizer.pad_token_id).all():
                    non_pad_mask = _out_ids == self.tokenizer.pad_token_id
                    last_non_pad = non_pad_mask.long().argmax(dim=-1).min().item()  # last non-padding index
                    _out_ids = _out_ids[:, : last_non_pad + 1]
                output_ids_to_save.append(_out_ids)
            if len(idxes_to_estimate):
                if self.n_samples_per_step > 1:
                    thoughts_to_estimate = torch.cat(thoughts_to_estimate, dim=0)
                    estimations = self.estimate(
                        batch_input_ids=input_ids_to_estimate,
                        batch_answers=answers_to_estimate,
                        latent_thoughts=thoughts_to_estimate,
                    )

                    assert (estimations >= 0).all(), f"estimations should be non-negative:\n{estimations}"
                else:
                    estimations = (
                        corrects[idxes_to_estimate]
                        .float()
                        .unsqueeze(-1)
                        .expand(-1, -1, self.num_latents)
                        .contiguous()
                    )
            if self.accelerator.is_main_process:
                _dict = dict(
                    pass_at_1=coverage / total_num_samples,
                    vot_acc=voting_accuracy / total_num_samples,
                    all_correct=all_correct_count / total_num_samples,
                    all_false=all_false_count / total_num_samples,
                )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        _dict | {"progress": batch_i * self.accelerator.num_processes / len(self.dataloader)}
                    )
                pbar.set_postfix({k: f"{v*100:.2f}%" for k, v in _dict.items()})

            for i, idx in enumerate(batch["idx"]):
                if i in idxes_to_estimate:
                    i_in_estimations = idxes_to_estimate.index(i)
                    if self.debug:
                        rich.print(
                            "*" * 100 + "\n",
                            f"inputs: {self.tokenizer.batch_decode(output_ids_to_save[i_in_estimations].contiguous(), skip_special_tokens=True)}",
                            "\n",
                            f"estimations: {estimations[i_in_estimations]}",
                            "\n",
                            f"cumulative estimations: {estimations[i_in_estimations].cumsum(dim=-1)}",
                            "\n",
                            f"corrects: {corrects[i]}",
                            "\n",
                            "*" * 100 + "\n\n",
                        )
                    self.results.update(
                        {
                            str(idx.item()) + ".input_ids": output_ids_to_save[i_in_estimations].contiguous(),
                            str(idx.item()) + ".estimations": estimations[i_in_estimations],
                            str(idx.item())
                            + ".latent_embeds": output.latent_thoughts[
                                i * self.n_samples : (i + 1) * self.n_samples, -self.num_latents :
                            ].contiguous(),
                            str(idx.item()) + ".corrects": corrects[i],
                        }
                    )
                    if self.save_answers:
                        self.results.update(
                            {
                                str(idx.item()) + ".answers": answer_output[i],
                            }
                        )
                else:
                    self.results.update(
                        {
                            str(idx.item()) + ".input_ids": torch.tensor(0, device="cpu"),
                        }
                    )
            del input_ids_list, output
            if len(self.results) >= self.save_freq:
                self.save_samples()
            gc.collect()
            torch.cuda.empty_cache()

        if len(self.results) > 0:
            self.save_samples()

        coverage_at_N = coverage / total_num_samples
        voting_accuracy_at_N = voting_accuracy / total_num_samples
        all_correct_rate = all_correct_count / total_num_samples
        all_false_rate = all_false_count / total_num_samples
        results = {
            "coverage_at_N": coverage_at_N,
            "voting_accuracy_at_N": voting_accuracy_at_N,
            "all_correct_rate": all_correct_rate,
            "all_false_rate": all_false_rate,
        }
        if self.accelerator.is_main_process:
            print("*" * 100)
            print(f"coverage@{self.n_samples}: {coverage_at_N * 100:.2f}% ({coverage}/{total_num_samples})")
            print(
                f"vot@{self.n_samples}: {voting_accuracy_at_N * 100:.2f}% ({voting_accuracy}/{total_num_samples})"
            )
            print(
                f"all correct rate: {all_correct_rate * 100:.2f}% ({all_correct_count}/{total_num_samples})"
            )
            print(f"all false rate: {all_false_rate * 100:.2f}% ({all_false_count}/{total_num_samples})")
            print("*" * 100)

            if self.wandb_run is not None:
                for k, v in results.items():
                    self.wandb_run.summary[k] = v
                self.wandb_run.finish()

        return results

    def estimate(
        self,
        batch_input_ids: list[torch.Tensor],
        batch_answers: list[str],
        latent_thoughts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the probability of the correct answer for each rollout for each sample in the batch.
        Note that we do not estimate the final step, but rather use the correctness of this sequence.

        Args:
            batch_input_ids: list[torch.Tensor], the input ids of the batch, in which padding ids are removed, (cur_B, L)
            batch_answers: list[str], the answers of the batch (cur_B,)
            latent_thoughts: torch.Tensor, the latent thoughts of the batch (cur_B * n_samples, L + num_latents, d)
        Returns:
            estimations: torch.Tensor, shape: (batch_size, n_samples, num_latents).
            estimations[i, j, k] is the probability of the correct answer for the j-th rollout of the i-th sample at the k-th step.

        """
        current_batch_size = len(batch_input_ids)

        def iterate():
            for i in range(current_batch_size):
                for j in range(self.n_samples):
                    for k in range(self.num_latents):
                        yield i, j, k

        _num_samples_to_take = self.batch_size // self.n_samples_per_step
        mini_batch = []
        latent_embeds = []
        # init with -1 to allow checking if every position is visited
        estimations = (
            torch.zeros(current_batch_size, self.n_samples, self.num_latents, device=self.model.device) - 1
        )
        # estimations[..., -1] = corrects.float().to(self.model.device)
        mask_travelled = torch.zeros(
            current_batch_size, self.n_samples, self.num_latents, device=self.model.device, dtype=torch.bool
        )
        if self.accelerator.is_main_process and self.batch_progress_bar:
            pbar = tqdm(
                iterate(),
                total=current_batch_size * self.n_samples * self.num_latents,
                desc="Estimating",
            )
        else:
            pbar = iterate()

        for input_i, sample_i, step_i in pbar:
            mask_travelled[input_i, sample_i, step_i] = True
            input_ids = torch.cat(
                [
                    batch_input_ids[input_i],
                    torch.full((step_i + 1,), self.model.config.latent_id, device=self.model.device),
                ],
                dim=0,
            )  # (L,)
            # num_latents = 6
            # latent_thoughts spands from [x0, x1, ..., latent0, latent1, latent2, latent3, latent4, latent5]
            # step_i = 0, _right = -5, _end = -5-5 [-5-5:-5]
            # step_i = 1, _right = -4, _end = -4-5 [-4-5:-4]
            # step_i = 2, _right = -3, _end = -3-5 [-3-5:-3]
            # step_i = 3, _right = -2, _end = -2-5 [-2-5:-2]
            # step_i = 4, _right = -1, _end = -1-5 [-1-5:-1]
            # step_i = ?, _right = - num_latents + 1 + step_i, _left = _right - num_latents
            _right = -self.num_latents + step_i + 1
            _left = _right - self.num_latents
            if _right == 0:
                _right = None
            selected_latent_embeds = latent_thoughts[input_i * self.n_samples + sample_i, _left:_right]
            latent_embeds.append(selected_latent_embeds)  # (num_latents, d)

            mini_batch.append(
                {
                    "input_ids": input_ids,
                    "answer": batch_answers[input_i],
                }
            )
            if len(mini_batch) == _num_samples_to_take:
                mini_batch = self.collator(mini_batch)
                answers = mini_batch["answer"]
                mini_batch = {
                    k: v.to(self.model.device)
                    for k, v in mini_batch.items()
                    if k in ["input_ids", "attention_mask"]
                }
                mini_batch["inputs_embeds"] = self.model.get_input_embeddings()(
                    torch.where(
                        mini_batch["input_ids"] != self.model.config.latent_id, mini_batch["input_ids"], 0
                    )
                )
                mini_batch["inputs_embeds"][:, -self.num_latents :] = torch.stack(latent_embeds, dim=0)
                output = self.model.generate(
                    **mini_batch,
                    generation_config=self.estimation_generation_config,
                    num_return_sequences=self.n_samples_per_step,
                )
                corrects = self.check_output(
                    output[:, mini_batch["input_ids"].shape[1] :], answers, self.n_samples_per_step
                )["corrects"]
                estimations[mask_travelled] = corrects.float().to(self.model.device).mean(dim=-1)
                mask_travelled.fill_(False)
                mini_batch = []
                latent_embeds = []

                del output, corrects
                gc.collect()
                torch.cuda.empty_cache()

        if len(mini_batch) > 0:
            mini_batch = self.collator(mini_batch)
            answers = mini_batch["answer"]
            mini_batch = {
                k: v.to(self.model.device)
                for k, v in mini_batch.items()
                if k in ["input_ids", "attention_mask"]
            }
            mini_batch["inputs_embeds"] = self.model.get_input_embeddings()(
                torch.where(
                    mini_batch["input_ids"] != self.model.config.latent_id, mini_batch["input_ids"], 0
                )
            )
            mini_batch["inputs_embeds"][:, -self.num_latents :] = torch.stack(latent_embeds, dim=0)
            output = self.model.generate(
                **mini_batch,
                generation_config=self.estimation_generation_config,
                num_return_sequences=self.n_samples_per_step,
            )
            corrects = self.check_output(
                output[:, mini_batch["input_ids"].shape[1] :], answers, self.n_samples_per_step
            )["corrects"]
            estimations[mask_travelled] = corrects.float().to(self.model.device).mean(dim=-1)
            mask_travelled.fill_(False)
            mini_batch = []
            latent_embeds = []

        return estimations

    def check_output(
        self,
        output,
        answers: list[str],
        n: int = 1,
        return_voting_result: bool = False,
        return_answer_output: bool = False,
    ) -> torch.Tensor:
        text_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        answer_output = [self.answer_extractor(text_output[i]) for i in range(len(text_output))]
        corrects = [answer_output[i] == answers[i // n] for i in range(len(answer_output))]
        corrects = torch.tensor(corrects).view(-1, n)
        results = {"corrects": corrects}
        if return_voting_result:
            voting_result = [
                Counter(answer_output[i : i + n]).most_common(1)[0][0]
                for i in range(0, len(answer_output), n)
            ]
            voting_result = [voting_result[i] == answers[i] for i in range(len(voting_result))]
            voting_result = torch.tensor(voting_result).view(-1)
            results["voting_result"] = voting_result
        if return_answer_output:
            results["answer_output"] = torch.tensor(answer_output).view(-1, n)
        return results

    def load_idxes(self, idx_file) -> list[int] | None:
        if idx_file == "":
            return None
        if os.path.exists(idx_file):
            print(f"Loading idxes from {idx_file}, ", end="")
            if idx_file.endswith(".pt"):
                idxes = torch.load(idx_file)
                idxes = list(set(idxes))
                print(f"{len(idxes)} idxes loaded")
            else:
                assert idx_file.endswith(
                    ".txt"
                ), f"idx_file must be a txt file or a pt file, but got {idx_file}"
                with open(idx_file, "r") as f:
                    idxes = [int(line.strip()) for line in f.readlines()]
                    idxes = list(set(idxes))
            print(f"{len(idxes)} idxes loaded")
        else:
            raise FileNotFoundError(f"File not found: {idx_file}")
        return idxes

    def set_dirs(self):
        # Set save directory
        self.save_dir = os.path.join(self.save_path, self.name)
        if not os.path.exists(self.save_dir) and self.accelerator.is_main_process:
            os.makedirs(self.save_dir)

    def set_tokenizer_model(self, generation_kwargs: dict = {}):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        latent_token, start_latent_token, end_latent_token = (
            "<|latent|>",
            "<|start-latent|>",
            "<|end-latent|>" if self.model_type != "colar" else "###",
        )
        # note colar do not have start-latent token
        latent_id = self.tokenizer.convert_tokens_to_ids(latent_token)
        start_id = self.tokenizer.convert_tokens_to_ids(start_latent_token)
        end_id = self.tokenizer.convert_tokens_to_ids(end_latent_token)
        self.explicit_start_id = self.tokenizer.convert_tokens_to_ids("<<")
        self.explicit_end_id = self.tokenizer.convert_tokens_to_ids(">>")
        assert (
            latent_id != start_id and latent_id != end_id
        ), f"latent_id {latent_id} is the same as start_id {start_id} or end_id {end_id}"

        class LatentLLM(self.model_class, LatentGenerationMixin):
            def __init__(self, config):
                super().__init__(config)

        self.model = LatentLLM.from_pretrained(
            self.model_id,
            latent_id=latent_id,
            latent_start_id=start_id,
            latent_end_id=end_id,
            pad_token_id=self.tokenizer.pad_token_id,
            device_map={"": self.accelerator.process_index},
        )

        if "latent_do_sample" in generation_kwargs:
            latent_do_sample = generation_kwargs["latent_do_sample"]
            if not latent_do_sample:
                warnings.warn("latent_do_sample is False, make sure this is expected")
            del generation_kwargs["latent_do_sample"]
        else:
            latent_do_sample = True

        do_latent_by_droput = (
            latent_do_sample and generation_kwargs.get("latent_do_sample_by", "") == "dropout"
        )

        if self.model_type == "colar" and do_latent_by_droput:
            find_and_replace_target_modules(
                self.model,
                target_modules=["down_proj"],
                target_module_type=torch.nn.Linear,
                replace_fn=MCLinear,
                p=generation_kwargs.get("dropout_p", 0.1),
            )

        self.generation_config = LatentGenerationConfig(
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            latent_do_sample=latent_do_sample,
            **generation_kwargs,
        )
        if do_latent_by_droput:
            warnings.warn(
                "latent_do_sample_by is dropout, setting explicit_do_sample and explicit_do_sample_by to dropout"
            )
            generation_kwargs["explicit_do_sample"] = True
            generation_kwargs["explicit_do_sample_by"] = "dropout"

        self.estimation_generation_config = LatentGenerationConfig(
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            latent_do_sample=latent_do_sample,
            **generation_kwargs,
        )

    def prepare_dataset(
        self,
        dataset_piece=1.0,
        dataset_indice=0,
        idx_file="",
        excude_idxes=False,
    ):
        """
        Process the data and return the dataset.
        """
        dataset = datasets.Dataset.from_json(self.data_path)
        if self.debug:
            dataset = dataset.select([234626])
            rich.print(f"[bold red]Selected {len(dataset)} samples for debug[/bold red]")
        dataset = dataset.map(
            lambda x, idx: {"idx": idx},
            with_indices=True,
        )

        _original_len = len(dataset)
        idxes = self.load_idxes(idx_file)
        if idxes is not None:
            all_idxes = set(dataset["idx"])
            idxes = set(idxes)
            if excude_idxes:
                rich.print(f"Deleting idxes in {idx_file}...", end="")
                idxes_to_select = all_idxes - idxes
                idxes_to_select = list(idxes_to_select)
                idxes_to_select.sort()
                dataset = dataset.select(idxes_to_select)
            else:
                rich.print(f"Selecting idxes in {idx_file}...", end="")
                dataset = dataset.select(idxes)
            assert len(dataset) > 0, f"dataset is empty after filtering"
            rich.print(
                f"removed {_original_len - len(dataset)} processed samples, remaining {len(dataset)} samples"
            )

        _original_len = len(dataset)
        if dataset_piece < 1.0:
            dataset_piece_size = int(_original_len * dataset_piece)
            num_pieces = math.ceil(_original_len / dataset_piece_size)
            last_piece_size = _original_len - dataset_piece_size * (num_pieces - 1)
            data_splits = [dataset_piece_size] * (num_pieces - 1) + [last_piece_size]

            average_piece_size = _original_len / num_pieces
            if num_pieces > 1 and last_piece_size < average_piece_size * 0.5:
                num_pieces -= 1
                data_splits[-2] += data_splits[-1]
                data_splits = data_splits[:-1]

            rich.print(f"Data splits: {data_splits}")
            assert (
                0 <= dataset_indice < num_pieces
            ), f"dataset_indice {dataset_indice} is out of range (should be < {num_pieces})"
            _begin = sum(data_splits[:dataset_indice])
            _end = _begin + data_splits[dataset_indice]
            dataset = dataset.select(range(_begin, _end))
            rich.print(f"Selected {_end - _begin} samples from the dataset")

        rich.print("Final dataset size: ", len(dataset))

        def convert_answer(x):
            try:
                return float(x["answer"].replace(",", ""))
            except:
                return False

        question_template = "Question: {} Let's think step by step:(Thinking speed: {})###"

        def convert_question(x):
            if self.model_type == "colar":
                return question_template.format(x["question"], 2)
            elif self.model_type == "codi":
                return x["question"] + "<|start-latent|>"
            else:
                return x["question"] + "\n<|start-latent|>"

        dataset = dataset.map(
            lambda x: {
                "question": convert_question(x),
                "answer": convert_answer(x),
                "to_remove": not convert_answer(x),
            },
        )
        _before_len = len(dataset)
        dataset = dataset.filter(lambda x: not x["to_remove"])
        if (removed := _before_len - len(dataset)) > 0:
            rich.print(f"Removed {removed} samples with complex answers")
        dataset = dataset.map(
            lambda x: self.tokenizer(x["question"], add_special_tokens=self.model_type != "colar"),
            batched=True,
        )

        dataset = dataset.map(lambda x: {"input_length": len(x["input_ids"])})
        dataset = dataset.sort("input_length", reverse=True)
        dataset = dataset.remove_columns("input_length")

        return dataset

    def save_samples(self):
        _tmp_path = os.path.join(
            self.save_dir, f"{self.num_saves}_on{self.accelerator.process_index}.safetensors"
        )
        save_file(self.results, _tmp_path)
        self.results = {}
        self.num_saves += 1


def run(**kwargs):

    with torch.no_grad():
        data_generator = DataGenerator(**kwargs)
        data_generator.run()


if __name__ == "__main__":
    import fire

    fire.Fire(run)
