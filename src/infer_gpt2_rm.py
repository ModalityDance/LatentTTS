from collections import Counter
from typing import Literal
import torch
import numpy as np
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from fire import Fire

from .generation_mixin import LatentGenerationMixin, LatentGenerationConfig
from .paths import MODELS
from .models.gpt2 import COCONUTGPT2ForTokenClassification
from .models.llama import COCONUTLlamaForTokenClassification
from .utils import set_seed, InferenceCollator


@torch.no_grad()
def main(
    generator_type: Literal["coconut", "codi"] = "coconut",
    model_dtype: Literal["bf16", "fp16", "fp32"] = "fp32",
    prm_id: str = "../../latentRM-coconut",
    prm_model_family: Literal["llama", "gpt2"] = "gpt2",
    prm_mode: Literal["best_of_n", "beam_search"] = "beam_search",
    data_path: str = "data/gsm_test.json",
    num_return_sequences: int = 1,
    num_beams: int = 4,
    num_beam_candidates: int = 4,
    batch_size: int = 1024,
    latent_length: int = 6,
    max_new_tokens: int | None = 128,
    seed: int = 200,
    sort_by_len: bool = True,
    progress_bar: bool = True,
):
    new_line_after_input = generator_type == "coconut"
    if prm_mode == "beam_search":
        assert num_return_sequences == 1
        batch_size = max(1, batch_size // num_beam_candidates // num_beams)
    else:
        assert num_return_sequences > 1
        num_beam_candidates = None
        num_beams = 1
        batch_size = max(1, batch_size // num_return_sequences)

    model_id = MODELS[generator_type]["id"]
    set_seed(seed)
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Loading {data_path}")

    processing_class = AutoTokenizer.from_pretrained(
        MODELS[generator_type]["id"],
    )
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    latent_id = processing_class.convert_tokens_to_ids("<|latent|>")
    start_id = processing_class.convert_tokens_to_ids("<|start-latent|>")
    end_id = processing_class.convert_tokens_to_ids("<|end-latent|>")
    target_id = processing_class.convert_tokens_to_ids(">>")
    assert (
        latent_id != start_id and latent_id != end_id and start_id != end_id
    ), f"latent_id, start_id, end_id must be different, but got {latent_id}, {start_id}, {end_id}"
    # print(f"<|latent|>: {latent_id}, <|start-latent|>: {start_id}, <|end-latent|>: {end_id}, >>: {target_id}")

    generation_config = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        generation_mode="beam_search" if prm_mode == "beam_search" else None,
        # do_sample=False,
        num_beams=num_beams,
        num_beam_candidates=num_beam_candidates,
        # num_samples=num_samples,
        pad_token_id=processing_class.pad_token_id,
        eos_token_id=processing_class.eos_token_id,
        bos_token_id=processing_class.bos_token_id,
    )

    class LatentGPT2LMHeadModel(MODELS[generator_type]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentGPT2LMHeadModel.from_pretrained(
        MODELS[generator_type]["id"],
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        target_id=target_id,
        pad_token_id=processing_class.pad_token_id,
        device_map={"": accelerator.process_index},
        torch_dtype=(
            torch.bfloat16 if model_dtype == "bf16" else (torch.float16 if model_dtype == "fp16" else None)
        ),
    )
    if prm_model_family == "llama":

        prm_processing_class = AutoTokenizer.from_pretrained(prm_id)
        prm = COCONUTLlamaForTokenClassification.from_pretrained(
            prm_id,
            latent_id=prm_processing_class.convert_tokens_to_ids("<|latent|>"),
            latent_start_id=prm_processing_class.convert_tokens_to_ids("<|start-latent|>"),
            latent_end_id=prm_processing_class.convert_tokens_to_ids("<|end-latent|>"),
            target_id=prm_processing_class.convert_tokens_to_ids(">>"),
            pad_token_id=prm_processing_class.pad_token_id,
            latent_hidden_size=model.config.hidden_size,
            device_map={"": accelerator.process_index},
            torch_dtype=(
                torch.bfloat16
                if model_dtype == "bf16"
                else (torch.float16 if model_dtype == "fp16" else None)
            ),
        )
    else:
        prm_processing_class = processing_class
        prm = COCONUTGPT2ForTokenClassification.from_pretrained(
            prm_id,
            latent_id=latent_id,
            latent_start_id=start_id,
            latent_end_id=end_id,
            target_id=target_id,
            pad_token_id=processing_class.pad_token_id,
            device_map={"": accelerator.process_index},
            torch_dtype=(
                torch.bfloat16
                if model_dtype == "bf16"
                else (torch.float16 if model_dtype == "fp16" else None)
            ),
        )
    prm.eval()
    model.eval()
    dataset = datasets.Dataset.from_json(data_path)
    postfix = "\n<|start-latent|>" if new_line_after_input else "<|start-latent|>"
    dataset = dataset.map(
        lambda x, idx: {
            "idx": idx,
            "question": x["question"] + postfix,
            "answer": float(x["answer"].replace(",", "")),
        },
        with_indices=True,
    )
    dataset = dataset.map(lambda x: processing_class(x["question"]), batched=True)
    if sort_by_len:
        dataset = dataset.map(lambda x: {"length": len(x["input_ids"])})
        dataset = dataset.sort("length")
        dataset = dataset.remove_columns("length")

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=InferenceCollator(processing_class))

    if prm_mode == "best_of_n":
        all_corrects = {}
        voting_accuracies = {}
    accuracies = {}

    if accelerator.is_main_process and progress_bar:
        pbar = tqdm(dataloader, colour="green", desc="Generating")
    else:
        pbar = dataloader

    for batch in pbar:
        model_inputs = {
            k: v.to(model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
        }

        output = model.generate(
            **model_inputs,
            process_reward_model=prm if prm_mode == "beam_search" else None,
            generation_config=generation_config,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            use_cache=True,
        )
        text_output = processing_class.batch_decode(output.sequences, skip_special_tokens=True)

        prm_input_texts = []
        if prm_model_family == "llama":
            for text in text_output:
                splited = text.split("\n")
                question = splited[0]
                answer = "\n".join(splited[1:])
                input_text = prm_processing_class.apply_chat_template(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    tokenize=False,
                )
                prm_input_texts.append(input_text)
            inputs = prm_processing_class(prm_input_texts, return_tensors="pt", padding=True)
        else:
            inputs = prm_processing_class(text_output, return_tensors="pt", padding=True)
        if prm_mode == "best_of_n":
            prm_scores = prm(
                input_ids=inputs["input_ids"].to(prm.device),
                attention_mask=inputs["attention_mask"].to(prm.device),
                latent_embeds=output.latent_thoughts.to(prm.device),
                return_dict=True,
            ).logits.squeeze(
                -1
            )  # (B * num_samples, seq)

            prm_scores = torch.where(
                (inputs["input_ids"] == prm.config.latent_id).to(prm.device), prm_scores, 0
            )

            prm_scores = prm_scores.sum(dim=-1)
            prm_scores = prm_scores.reshape(-1, num_return_sequences)
            top_indices = torch.topk(prm_scores, k=1, dim=1)[1]
            del prm_scores
        answer_output = [
            MODELS[generator_type]["answer_extractor"](text_output[i]) for i in range(len(text_output))
        ]
        if prm_mode == "best_of_n":
            voting_result = [
                Counter(answer_output[i : i + num_return_sequences]).most_common(1)[0][0]
                for i in range(0, len(answer_output), num_return_sequences)
            ]
            voting_result = [voting_result[i] == batch["answer"][i] for i in range(len(voting_result))]

        correct = [
            answer_output[i] == batch["answer"][i // num_return_sequences] for i in range(len(answer_output))
        ]
        correct = torch.tensor(correct).view(-1, num_return_sequences)
        del output, inputs
        torch.cuda.empty_cache()
        for i, _idx in enumerate(batch["idx"]):

            if prm_mode == "best_of_n":
                all_corrects[_idx.item()] = correct[i].tolist()
                voting_accuracies[_idx.item()] = voting_result[i]
                accuracies[_idx.item()] = correct[i][top_indices[i].item()]
            else:
                accuracies[_idx.item()] = correct.item()  # only one item since num_return_sequences == 1

        if accelerator.is_main_process and progress_bar:
            _corrects = torch.tensor(list(accuracies.values())).long().sum()
            cur_num_samples = len(accuracies)
            if prm_mode == "best_of_n":
                _cov = torch.tensor([v.any().long() for v in all_corrects.values()]).long().sum()
                _voting = torch.tensor(list(voting_accuracies.values())).long().sum()
                _dict = dict(
                    Cov=f"{_cov/cur_num_samples*100:.4f}% ({_cov}/{cur_num_samples})",
                    Acc=f"{_corrects/cur_num_samples*100:.4f}% ({_corrects}/{cur_num_samples})",
                    Vot=f"{_voting/cur_num_samples*100:.4f}% ({_voting}/{cur_num_samples})",
                )
            else:
                _dict = dict(
                    Acc=f"{_corrects/cur_num_samples*100:.4f}% ({_corrects}/{cur_num_samples})",
                )
            pbar.set_postfix(_dict)

    accelerator.wait_for_everyone()
    if accelerator.num_processes > 1:
        accuracies = accelerator.gather_for_metrics([accuracies], use_gather_object=True)
        accuracies = {idx: value for d in accuracies for idx, value in d.items()}
        if prm_mode == "best_of_n":
            all_corrects = accelerator.gather_for_metrics([all_corrects], use_gather_object=True)
            all_corrects = {idx: value for d in all_corrects for idx, value in d.items()}
            voting_accuracies = accelerator.gather_for_metrics([voting_accuracies], use_gather_object=True)
            voting_accuracies = {idx: value for d in voting_accuracies for idx, value in d.items()}

    if accelerator.is_main_process:
        print(f"SEED={seed}")

        corrects = np.array(list(accuracies.values()))
        print(f"Accuracy: {corrects.mean()*100:.4f}%")
        if prm_mode == "best_of_n":
            all_corrects = np.array(list(all_corrects.values()))
            coverages = all_corrects.any(axis=-1).float().mean()
            voting_accuracies = np.array(list(voting_accuracies.values()))
            print(f"Coverage: {coverages*100:.4f}%")
            print(f"Voting Accuracy: {voting_accuracies.mean()*100:.4f}%")


if __name__ == "__main__":
    Fire(main)
