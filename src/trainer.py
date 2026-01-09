from typing import Callable, Optional, Literal, Any
from functools import partial
from dataclasses import dataclass, field
import os

import numpy as np
import torch
import rich

from torch.utils.data import DataLoader, Dataset
from transformers import EvalPrediction, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import EvalLoopOutput, seed_worker
from transformers.utils import is_peft_available
from trl.trainer.utils import disable_dropout_in_model

from src.dataset import (
    CachedPickleDatasetV2,
    DataCollatorForLatentRM,
    DataCollatorForContrastiveLatentRM,
)
from src.models.gpt2 import COCONUTGPT2ForTokenClassification, CODIGPT2ForTokenClassification, CODIGPT2Config
from src.models.llama import COCONUTLlamaForTokenClassification
from src.models.loss import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss

if is_peft_available():
    from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class LatentRMConfig(TrainingArguments):


    average_tokens_across_devices: bool | None = field(
        default=True,
        metadata={
            "help": "Whether or not to average tokens across devices. If enabled, will use all_reduce to synchronize "
            "num_tokens_in_batch for precise loss calculation. Reference: https://github.com/huggingface/transformers/issues/34242 "
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    model_id: str = field(
        default="checkpoints/coconut",
        metadata={"help": "The id of the model to train."},
    )
    model_type: Literal["coconut", "codi"] = field(
        default="coconut",
        metadata={"help": "The type of the model to train."},
    )
    model_family: Literal["gpt2", "llama"] = field(
        default="gpt2",
        metadata={"help": "The family of the model to train."},
    )
    train_dataset: str = field(default="", metadata={"help": "The path to the dataset."})
    valid_dataset: dict[str, str] = field(
        default_factory=lambda: {},
        metadata={"help": "The path to the dataset or a dict of dataset names and paths"},
    )
    eval_on_start: bool = field(
        default=True,
        metadata={"help": "Whether to evaluate the model on the start of training."},
    )
    eval_strategy: Literal["steps", "epoch"] = field(
        default="steps",
        metadata={"help": "The strategy to use for evaluation."},
    )
    save_strategy: Literal["steps", "epoch"] = field(
        default="steps",
        metadata={"help": "The strategy to use for saving the model."},
    )
    eval_frequency: int = field(
        default=4,
        metadata={"help": "Number of evaluation to be done per epoch."},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether to load the best model at the end of training."},
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "The report to use for reporting the metrics."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Whether to find unused parameters in the DDP."},
    )
    gradient_checkpointing_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={"help": "The gradient checkpointing kwargs to use for training."},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing."},
    )
    metric_for_best_model: str = field(
        default="eval_f1",
        metadata={"help": "The metric to use for the best model."},
    )
    include_for_metrics: list[str] = field(
        default_factory=lambda: ["inputs"],
        metadata={"help": "The metrics to include for the evaluation."},
    )
    early_stopping_patience: int = field(
        default=6,
        metadata={"help": "The patience to use for early stopping."},
    )
    latent_hidden_size: int = field(
        default=768,
        metadata={
            "help": "The hidden size of the latent thoughts. If not specified, it is the hidden size of the gpt2 model."
        },
    )
    generator_model_id: str = field(
        default="checkpoints/coconut",
        metadata={"help": "The id of the coconut model."},
    )
    loss_type: Literal["bce", "ce"] = field(
        default="bce",
        metadata={"help": "The loss type to use for training."},
    )



class LatentRMTrainer(Trainer):
    _tag_names = ["trl"]

    def __init__(
        self,
        args: LatentRMConfig,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict[str, Any] | PeftConfig] = None,
    ):

        processing_class = AutoTokenizer.from_pretrained(args.model_id)
        processing_class.pad_token = processing_class.eos_token

        latent_id = processing_class.convert_tokens_to_ids("<|latent|>")
        start_id = processing_class.convert_tokens_to_ids("<|start-latent|>")
        end_id = processing_class.convert_tokens_to_ids("<|end-latent|>")
        if latent_id is None:
            processing_class.add_tokens("<|latent|>")
            latent_id = processing_class.convert_tokens_to_ids("<|latent|>")
            rich.print(f"Added <|latent|> token : {latent_id}")
        if start_id is None:
            processing_class.add_tokens("<|start-latent|>")
            start_id = processing_class.convert_tokens_to_ids("<|start-latent|>")
            rich.print(f"Added <|start-latent|> token : {start_id}")
        if end_id is None:
            processing_class.add_tokens("<|end-latent|>")
            end_id = processing_class.convert_tokens_to_ids("<|end-latent|>")
            rich.print(f"Added <|end-latent|> token : {end_id}")
        assert (
            latent_id != start_id and latent_id != end_id
        ), f"latent_id: {latent_id}, start_id: {start_id}, end_id: {end_id}"
        rich.print(f"<|latent|>: {latent_id}, <|start-latent|>: {start_id}, <|end-latent|>: {end_id}")
        ### set dataset ###

        if args.train_dataset:
            train_dataset = CachedPickleDatasetV2(
                data_dir=args.train_dataset,
                verbose=False,
                include_gt=False,
                get_single_sample=args.loss_type == "bce",
            )
            if args.loss_type == "ce":
                args.per_device_train_batch_size = args.per_device_train_batch_size // train_dataset.n_samples
        else:
            train_dataset = None

        eval_dataset = {
            name: CachedPickleDatasetV2(
                data_dir=dataset,
                verbose=False,
                include_gt=True,
                get_single_sample=True,
            )
            for name, dataset in args.valid_dataset.items()
        }
        for name, dataset in eval_dataset.items():
            rich.print(name, dataset)

        self._under_eval_dataset = None
        ### set model ###

        if args.model_family == "gpt2":
            if args.model_type == "coconut":
                model = COCONUTGPT2ForTokenClassification.from_pretrained(
                    args.model_id,
                    latent_id=latent_id,
                    latent_start_id=start_id,
                    latent_end_id=end_id,
                )
            else:
                codi_config = CODIGPT2Config.from_pretrained(args.model_id)
                codi_config.latent_id = latent_id
                codi_config.latent_start_id = start_id
                codi_config.latent_end_id = end_id
                model = CODIGPT2ForTokenClassification.from_pretrained(
                    args.model_id,
                    config=codi_config,
                )

        elif args.model_family == "llama":
            model = COCONUTLlamaForTokenClassification.from_pretrained(
                args.model_id,
                latent_id=latent_id,
                latent_start_id=start_id,
                latent_end_id=end_id,
                latent_hidden_size=args.latent_hidden_size,
            )
            if model.get_input_embeddings().weight.shape[0] != len(processing_class):
                rich.print(
                    f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(processing_class)}"
                )
                model.resize_token_embeddings(len(processing_class))
        else:
            raise ValueError(f"Model family {args.model_family} not supported.")

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )

        if is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if args.gradient_checkpointing_kwargs is not None:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            model = get_peft_model(
                model, peft_config if isinstance(peft_config, PeftConfig) else PeftConfig(**peft_config)
            )

        if args.disable_dropout:
            disable_dropout_in_model(model)

        ### set loss function ###

        if args.loss_type == "bce":
            self.loss_fct = MaskedBCEWithLogitsLoss()
        elif args.loss_type == "ce":
            self.loss_fct = MaskedCrossEntropyLoss()
        ### set data collator ###


        data_collator_class = (
            DataCollatorForLatentRM if args.loss_type == "bce" else DataCollatorForContrastiveLatentRM
        )
        data_collator = data_collator_class(
            processing_class,
            latent_token_id=latent_id,
            latent_end_id=end_id,
            remove_pad_token=True,
            generator_tokenizer=(
                AutoTokenizer.from_pretrained(args.generator_model_id)
                if args.model_family != "gpt2"
                else None
            ),
        )
        self.eval_data_collator = DataCollatorForLatentRM(
            processing_class,
            latent_token_id=latent_id,
            remove_pad_token=True,
            generator_tokenizer=(
                AutoTokenizer.from_pretrained(args.generator_model_id)
                if args.model_family != "gpt2"
                else None
            ),
        )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=self.compute_loss_func,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # num_processes = self.accelerator.num_processes
        # num_batches = len(train_dataset) // self.args.per_device_train_batch_size // num_processes
        # eval_steps = num_batches // self.args.eval_frequency
        self.args.eval_steps = 1 / self.args.eval_frequency / self.args.num_train_epochs
        self.args.save_steps = 1 / self.args.eval_frequency / self.args.num_train_epochs

    # def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
    #     if train_dataset is None:
    #         train_dataset = self.train_dataset

    #     return RandomSampler(train_dataset.weights)

    def compute_loss_func(self, outputs, labels, num_items_in_batch=None):
        logits = outputs.logits
        if self.args.loss_type == "bce":
            return self.loss_fct(logits, labels)
        elif self.args.loss_type == "ce":
            if self._under_eval_dataset is None:
                n_samples = self.train_dataset.n_samples
            else:
                n_samples = self._under_eval_dataset.n_samples
            return self.loss_fct(logits, labels, n_samples=n_samples)

    def train(self):
        super().train()
        if self.args.load_best_model_at_end:
            best_dir = os.path.join(self.args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            self.save_model(best_dir)
            print(f"Saved best model to {best_dir}")

    def compute_sequence_metrics(
        self,
        scores: torch.FloatTensor,
        labels: torch.LongTensor,
        inputs: torch.LongTensor,
        filter_mask: torch.BoolTensor,
    ) -> dict[str, float]:
        current_eval_dset = self._under_eval_dataset
        if self.args.loss_type == "bce":
            scores = torch.where(filter_mask, scores, 1)  # (B, S)
            seq_scores = scores.log().sum(dim=-1)  # (B,)
        elif self.args.loss_type == "ce":
            scores = torch.where(filter_mask, scores, 0)  # (B, S)
            seq_scores = scores.sum(dim=-1)  # (B,)
        num_test_samples = current_eval_dset.num_test_samples
        tokenized_inputs = self.processing_class.batch_decode(inputs, skip_special_tokens=True)  # (B,)
        tokenized_inputs = [tokens.split("<|latent|>")[0] for tokens in tokenized_inputs]
        samples_list = list(set(tokenized_inputs))
        assert (
            len(samples_list) == num_test_samples
        ), f"len(samples_list): {len(samples_list)}, num_test_samples: {num_test_samples}"

        results = [[] for _ in range(len(samples_list))]
        sequence_labels = [[] for _ in range(len(samples_list))]
        for data_i in range(scores.shape[0]):
            tokenized_input = tokenized_inputs[data_i]
            indice = samples_list.index(tokenized_input)
            seq_score = seq_scores[data_i]
            label = labels[data_i]
            label = label[label != -100][-1]  # the ground truth at the final token
            results[indice].append(seq_score.item())
            sequence_labels[indice].append(label.item())

        results = torch.tensor(results)  # (N, S)
        sequence_labels = torch.tensor(sequence_labels, dtype=torch.long)  # (N, S)

        pos_mask = sequence_labels == 1  # (N, S) bool
        neg_mask = ~pos_mask  # (N, S) bool

        pos_sum = (results * pos_mask).sum(dim=1)  # (N,)
        neg_sum = (results * neg_mask).sum(dim=1)  # (N,)
        pos_cnt = pos_mask.sum(dim=1)  # (N,)
        neg_cnt = neg_mask.sum(dim=1)  # (N,)
        pos_mean = torch.where(pos_cnt > 0, pos_sum / pos_cnt.clamp(min=1), 0)
        neg_mean = torch.where(neg_cnt > 0, neg_sum / neg_cnt.clamp(min=1), 0)

        diff = pos_mean - neg_mean

        if (num_add_samples := current_eval_dset.skipped_num_tests) > 0:
            # the testset here is not full, since we excluded the all-true samples in the testsets
            # so we add some all-true samples to the testset
            results = torch.cat([results, torch.ones(num_add_samples, results.shape[1])], dim=0)
            sequence_labels = torch.cat(
                [sequence_labels, torch.ones(num_add_samples, sequence_labels.shape[1])], dim=0
            )
        assert (
            results.shape[1] == current_eval_dset.n_samples
        ), f"results.shape[1]: {results.shape[1]}, current_eval_dset.n_samples: {current_eval_dset.n_samples}"
        # 然后这就变成了一个ranking task
        _, recall_at_1 = self.rank_at_k(results, sequence_labels, 1)
        metrics = {"recall_at_1": recall_at_1}
        if current_eval_dset.n_samples > 8:
            _, recall_at_8 = self.rank_at_k(results, sequence_labels, 8)
            metrics["recall_at_8"] = recall_at_8

        return metrics

    @staticmethod
    def rank_at_k(results: torch.Tensor, labels: torch.Tensor, k: int) -> tuple[float, float]:
        topk_idx = results.topk(k, dim=1).indices  # (N, k)
        sorted_labels = torch.gather(labels, 1, topk_idx)  # (N, k)

        # ---- NDCG ----
        gains = 2 ** sorted_labels.float() - 1
        discounts = torch.log2(torch.arange(k, device=results.device).float() + 2)
        dcg = (gains / discounts).sum(dim=1)
        ideal_sorted_labels, _ = labels.sort(dim=1, descending=True)
        ideal_topk = ideal_sorted_labels[:, :k]
        ideal_gains = 2 ** ideal_topk.float() - 1
        ideal_dcg = (ideal_gains / discounts).sum(dim=1)
        ndcg = torch.where(ideal_dcg > 0, dcg / ideal_dcg, torch.zeros_like(dcg))

        # ---- Recall ----
        recalls = sorted_labels[:, :k].float().sum(dim=1)
        recall = torch.where(ideal_dcg > 0, recalls, torch.zeros_like(recalls))
        return ndcg.mean().item(), recall.mean().item()

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]:
        predictions: np.ndarray = eval_pred.predictions  # (B, S, D)
        predictions = torch.from_numpy(predictions)
        labels: np.ndarray = eval_pred.label_ids  # (B, S)
        labels = torch.from_numpy(labels)
        filter_mask = labels.long() != -100  # remove ignored tokens (-100)
        labels[labels != -100] = (labels[labels != -100] > 0).float()
        inputs: np.ndarray = eval_pred.inputs  # (B, S)
        inputs = torch.from_numpy(inputs)
        inputs[inputs == -100] = self.processing_class.pad_token_id
        if self.args.loss_type == "bce":
            scores = predictions.sigmoid().squeeze(-1)
            predictions = predictions > 0.5
            predictions = predictions.squeeze(-1)
        elif self.args.loss_type == "ce":
            scores = predictions.squeeze(-1)
        else:
            raise ValueError(f"Invalid loss type: {self.args.loss_type}")

        return self.compute_sequence_metrics(scores, labels, inputs, filter_mask)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self._under_eval_dataset = dataloader.dataset
        result = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        self._under_eval_dataset = None
        return result

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.eval_data_collator if not is_training else self.data_collator

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return self.accelerator.prepare(dataloader)
