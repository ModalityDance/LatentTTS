import gc
from collections import Counter
from typing import Literal

import torch
import numpy as np
from tqdm import tqdm
import datasets

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from fire import Fire

from .generation_mixin import LatentGenerationMixin, LatentGenerationConfig
from .models.perturbation import find_and_replace_target_modules, MCLinear
from .paths import MODELS
from .utils import pass_at_k_mean, InferenceCollator


@torch.no_grad()
def main(
    model_type: Literal["colar"] = "colar",
    model_dtype: Literal["bf16", "fp16", "fp32"] = "fp32",
    data_path: str = "data/gsm_test.json",
    n_samples: int = 1,
    batch_size: int = 512,
    max_latent_length: int = 64,
    max_new_tokens: int | None = 128,
    do_sample: bool = True,
    sampling_by: Literal["dropout", "noise"] = "dropout",
    noise_std: float | None = None,
    dropout_p: float | None = 0.1,
):
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Loading {data_path}")

    class LatentLLM(MODELS[model_type]["class"], LatentGenerationMixin): ...

    processing_class = AutoTokenizer.from_pretrained(
        MODELS[model_type]["id"],
    )
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token
    print(f"pad_token: {processing_class.pad_token} ({processing_class.pad_token_id})")

    latent_id = processing_class.convert_tokens_to_ids("<|latent|>")
    print(f"<|latent|>: {latent_id}")

    start_id = -1
    end_id = processing_class.convert_tokens_to_ids("###")

    generation_config = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=None,
        max_latent_length=max_latent_length,
        latent_do_sample=do_sample,
        latent_do_sample_by=sampling_by if do_sample else None,
        explicit_do_sample=do_sample,
        explicit_do_sample_by=sampling_by if do_sample else None,
        noise_std=noise_std,
        dropout_p=dropout_p,
        pad_token_id=processing_class.pad_token_id,
        eos_token_id=processing_class.eos_token_id,
        bos_token_id=processing_class.bos_token_id,
    )

    model = LatentLLM.from_pretrained(
        MODELS[model_type]["id"],
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        pad_token_id=processing_class.pad_token_id,
        device_map={"": accelerator.process_index},
        torch_dtype=(
            torch.bfloat16 if model_dtype == "bf16" else (torch.float16 if model_dtype == "fp16" else None)
        ),
    )
    if do_sample and sampling_by == "dropout":
        find_and_replace_target_modules(
            model,
            target_modules=["down_proj"],
            target_module_type=torch.nn.Linear,
            replace_fn=MCLinear,
            p=dropout_p,
        )
    # rich.print(model)

    dataset = datasets.Dataset.from_json(data_path)
    question_template = "Question: {} Let's think step by step:(Thinking speed: 2)###"
    dataset = dataset.map(
        lambda x, idx: {
            "idx": idx,
            "question": question_template.format(x["question"]),
            "answer": float(x["answer"].replace(",", "")),
        },
        with_indices=True,
    )
    dataset = dataset.map(
        lambda x: processing_class(x["question"], add_special_tokens=False),
        batched=True,
    )

    dataset = dataset.map(lambda x: {"length": len(x["input_ids"])})
    dataset = dataset.sort("length")
    dataset = dataset.remove_columns("length")

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=InferenceCollator(processing_class))

    corrects = {}
    accuracies = {}
    if accelerator.is_main_process:
        pbar = tqdm(dataloader, colour="green", desc="Generating")
    else:
        pbar = dataloader
    for batch in pbar:
        model_inputs = {
            k: v.to(model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
        }

        output = model.generate(
            **model_inputs,
            generation_config=generation_config,
            num_return_sequences=n_samples,
            use_cache=True,
        )

        text_output = processing_class.batch_decode(output, skip_special_tokens=True)

        answer_output = [
            MODELS[model_type]["answer_extractor"](text_output[i]) for i in range(len(text_output))
        ]
        voting_result = [
            Counter(answer_output[i : i + n_samples]).most_common(1)[0][0]
            for i in range(0, len(answer_output), n_samples)
        ]
        voting_result = [voting_result[i] == batch["answer"][i] for i in range(len(voting_result))]
        correct = [answer_output[i] == batch["answer"][i // n_samples] for i in range(len(answer_output))]

        correct = torch.tensor(correct).view(-1, n_samples)
        for i, _idx in enumerate(batch["idx"]):
            corrects[_idx.item()] = correct[i].tolist()
            accuracies[_idx.item()] = voting_result[i]

        del (
            model_inputs,
            output,
            text_output,
            answer_output,
            voting_result,
            correct,
        )
        gc.collect()
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.num_processes > 1:
        corrects = accelerator.gather_for_metrics([corrects], use_gather_object=True)
        corrects = {idx: value for d in corrects for idx, value in d.items()}

        accuracies = accelerator.gather_for_metrics([accuracies], use_gather_object=True)
        accuracies = {idx: value for d in accuracies for idx, value in d.items()}

    if accelerator.is_main_process:

        corrects = np.array(list(corrects.values()))

        root_n_samples = int(np.sqrt(n_samples))
        ks = [2**i for i in range(root_n_samples + 1) if 2**i <= n_samples]
        pass_at_k_scores = [pass_at_k_mean(corrects, k) for k in ks]
        for k, score in zip(ks, pass_at_k_scores):
            print(f"Pass@{k}: {score*100:.4f}")

        accuracies = np.array(list(accuracies.values()))
        accuracy = accuracies.mean()
        print(f"Voting Accuracy: {accuracy*100:.4f}")


if __name__ == "__main__":
    Fire(main)
