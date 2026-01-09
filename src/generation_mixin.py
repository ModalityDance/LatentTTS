import inspect
import warnings
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Any, Union, Literal, Dict, Tuple

import torch
import torch.nn as nn
from transformers import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.utils.import_utils import is_torchdynamo_compiling
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.utils import (
    GenerateOutput,
    GenerateDecoderOnlyOutput,
)

from transformers.cache_utils import (
    Cache,
    StaticCache,
    DynamicCache,
)
from transformers.utils import logging

from src.utils import enable_dropout, disable_dropout, add_noise, set_dropout_p


logger = logging.get_logger(__name__)


@dataclass
class LatentGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of decoder-only latent generation models, when using non-beam methods.
    """

    sequences: torch.LongTensor
    sequences_embeds: Optional[torch.FloatTensor] = None
    top_indices: Optional[torch.LongTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    latent_thoughts: Optional[Tuple[torch.FloatTensor]] | None = None


class LatentGenerationConfig(GenerationConfig):
    latent_length: int | None = None
    max_latent_length: int | None = None
    latent_do_sample: bool = False
    explicit_do_sample: bool = False
    latent_do_sample_by: Literal["dropout", "noise"] | None = None
    explicit_do_sample_by: (
        Literal["dropout", "noise", "vocab"] | None
    ) = None
    noise_std: float = 0.1
    dropout_p: float | None = None
    num_beam_candidates: int | None = None
    latent_pooling: Literal["mean", "max"] = "mean"
    concat_inputs: bool = False


    def __init__(self, **kwargs):
        kwargs["do_sample"] = (
            kwargs.get("do_sample", False)
            or kwargs.get("latent_do_sample", False)
            or kwargs.get("explicit_do_sample", False)
        )
        if kwargs.get("generation_mode", "") == "beam_search":
            kwargs["do_sample"] = False
        latent_do_sample = kwargs.get("latent_do_sample", False)
        explicit_do_sample = kwargs.get("explicit_do_sample", False)
        latent_do_sample_by = kwargs.get("latent_do_sample_by", "")
        explicit_do_sample_by = kwargs.get("explicit_do_sample_by", "")

        if latent_do_sample and not latent_do_sample_by:
            raise ValueError("latent_do_sample is True, but latent_do_sample_by is not provided")
        if explicit_do_sample and not explicit_do_sample_by:
            raise ValueError("explicit_do_sample is True, but explicit_do_sample_by is not provided")

        if not latent_do_sample and latent_do_sample_by != "":
            warnings.warn("latent_do_sample is False, setting latent_do_sample_by to None")
            kwargs["latent_do_sample_by"] = None
            latent_do_sample_by = ""

        if not explicit_do_sample and explicit_do_sample_by != "":
            warnings.warn("explicit_do_sample is False, setting explicit_do_sample_by to None")
            kwargs["explicit_do_sample_by"] = None
            explicit_do_sample_by = ""

        if kwargs.get("dropout_p", None) is not None:
            not_latent_dropout = not latent_do_sample or latent_do_sample_by != "dropout"
            not_explicit_dropout = not explicit_do_sample or latent_do_sample_by != "dropout"
            if not_latent_dropout and not_explicit_dropout:
                warnings.warn(
                    "dropout_p is provided, but latent_dropout and explicit_dropout are not activated, setting dropout_p to None"
                )
                kwargs["dropout_p"] = None

        num_beam_candidates = kwargs.get("num_beam_candidates", None)
        if not kwargs.get("generation_mode", "") == "beam_search" and not num_beam_candidates is None:
            warnings.warn("num_beam_candidates must be None")
            kwargs["num_beam_candidates"] = None

        if (max_latent_length := kwargs.get("max_latent_length", None)) is not None:
            assert max_latent_length < kwargs.get(
                "max_new_tokens", float("inf")
            ), "max_latent_length must be less than max_new_tokens"
        if (latent_length := kwargs.get("latent_length", None)) is not None:
            assert latent_length < kwargs.get(
                "max_new_tokens", float("inf")
            ), "latent_length must be less than max_new_tokens"
        assert (
            max_latent_length is not None or latent_length is not None
        ), "max_latent_length or latent_length must be provided"
        assert (
            max_latent_length is None or latent_length is None
        ), "max_latent_length and latent_length cannot be provided at the same time"

        super().__init__(**kwargs)



class LatentGenerationMixin(GenerationMixin):
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *contrastive search* if `penalty_alpha>0` and `top_k>1`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-search decoding* if `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder

        input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        return inputs, input_name, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[LatentGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        process_reward_model: Optional[Callable] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        if self.config.latent_id in model_kwargs["input_ids"]:
            if "inputs_embeds" not in model_kwargs:
                raise ValueError("Found latent_id in input_ids, but inputs_embeds is not provided")
            assert (
                model_kwargs["inputs_embeds"].shape[:2] == model_kwargs["input_ids"].shape[:2]
            ), "inputs_embeds and input_ids must have the same batch size and sequence length"
        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (
                is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
            ) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = True
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )
        else:
            raise ValueError("Only decoder-only models are supported for continuous generation.")

        # generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        assert (
            not generation_config.token_healing
        ), "Token healing is not supported for continuous generation."
        assert streamer is None, "Streamer is not supported for continuous generation."

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        # tocheck
        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, None, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode()

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        if generation_config.dropout_p is not None:
            set_dropout_p(self, generation_config.dropout_p)

        # 10. go into different generation modes, not ASSISTED_GENERATION, not DOLA_GENERATION
        assert generation_mode in (
            GenerationMode.SAMPLE,
            GenerationMode.GREEDY_SEARCH,
            GenerationMode.BEAM_SEARCH,
        ), f"Only greedy search, sample, beam search are supported for continuous generation, but got {generation_mode}"
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        if generation_mode == GenerationMode.SAMPLE or generation_mode == GenerationMode.GREEDY_SEARCH:
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=False,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SEARCH:
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=False,
                **model_kwargs,
            )
            result = self._beam_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                process_reward_model=process_reward_model,
                **model_kwargs,
            )
        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True  # Should check for `True` after v4.47
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def check_if_latent_sequence(
        self, input_ids, num_latents=None, max_num_latents=None
    ) -> (torch.BoolTensor, torch.BoolTensor):
        """
        Returns:
            latent_sequences: whether the next token should be <|latent|>
            latent_sequences_end: whether the next token should be <|latent_end|>
        """
        batch_size, seq_len = input_ids.shape
        assert (
            num_latents is not None or max_num_latents is not None
        ), "num_latents or max_num_latents must be provided"
        assert (
            num_latents is None or max_num_latents is None
        ), "num_latents and max_num_latents cannot be provided at the same time"
        if num_latents is not None:
            # 0. long enough
            if seq_len > num_latents:
                # 1. the last num_latents are all <|latent|>
                latent_sequences_end = (input_ids[:, -num_latents:] == self.config.latent_id).all(dim=-1)
            else:
                latent_sequences_end = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            # 2. <|latent_end|> is not in the sequence
            condition2 = torch.logical_and(
                ~latent_sequences_end, ~(input_ids == self.config.latent_end_id).any(dim=-1)
            )
            # 3. the last token is <|latent_start|> or <|latent|>
            condition3 = torch.logical_or(
                input_ids[:, -1] == self.config.latent_start_id, input_ids[:, -1] == self.config.latent_id
            )

            latent_sequences = torch.logical_and(condition2, condition3)
        elif max_num_latents is not None:
            # 1. there are less than max_num_latents tokens that are <|latent|>
            latent_must_stop = (input_ids[:, -max_num_latents:] == self.config.latent_id).all(dim=-1)
            # 2. <|latent_end|> is not in the sequence
            latent_ended = (input_ids == self.config.latent_end_id).long().sum(dim=-1) > 1
            latent_sequences = torch.logical_and(~latent_must_stop, ~latent_ended)

            latent_sequences_end = None
        return latent_sequences, latent_sequences_end

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: LatentGenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        input_embeds: Optional[torch.FloatTensor] = None,
        process_reward_model: Optional[Callable] = None,
        **model_kwargs,
    ) -> Union[LatentGenerateDecoderOnlyOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        explicit_do_sample = generation_config.explicit_do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        is_prefill = True

        if "inputs_embeds" not in model_kwargs:
            if input_embeds is None:
                model_kwargs["inputs_embeds"] = self.get_input_embeddings()(
                    torch.where(input_ids == self.config.latent_id, 0, input_ids)
                )
            else:
                model_kwargs["inputs_embeds"] = input_embeds

        dropout_on = False

        explicit_do_sample_with_dropout = (
            generation_config.explicit_do_sample_by == "dropout"
        ) and generation_config.explicit_do_sample
        latent_do_sample_with_dropout = (
            generation_config.latent_do_sample_by == "dropout"
        ) and generation_config.latent_do_sample

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(**model_kwargs)

            # check if we should do latent generation
            latent_sequences, latent_sequences_end = self.check_if_latent_sequence(
                input_ids,
                num_latents=generation_config.latent_length,
                max_num_latents=generation_config.max_latent_length,
            )

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            if latent_do_sample_with_dropout and not explicit_do_sample_with_dropout:
                if not (latent_sequences.all() or (~latent_sequences).all()):
                    raise ValueError("latent_sequences should be all True or all False")

            if latent_sequences.any():
                if latent_do_sample_with_dropout and not dropout_on:
                    enable_dropout(self)
                    dropout_on = True
                elif not latent_do_sample_with_dropout and dropout_on:
                    disable_dropout(self)
                    dropout_on = False
            else:
                if explicit_do_sample_with_dropout and not dropout_on:
                    enable_dropout(self)
                    dropout_on = True
                elif not explicit_do_sample_with_dropout and dropout_on:
                    disable_dropout(self)
                    dropout_on = False

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True, output_hidden_states=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True, output_hidden_states=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_scores:
                    scores += (next_token_scores.softmax(dim=-1),)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # token selection
            if explicit_do_sample and generation_config.explicit_do_sample_by == "vocab":
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids
            if latent_sequences_end is not None:
                next_tokens = torch.where(latent_sequences, self.config.latent_id, next_tokens)
                next_tokens = torch.where(latent_sequences_end, self.config.latent_end_id, next_tokens)
            else:
                latent_sequences_end = next_tokens == self.config.latent_end_id
                latent_sequences = torch.logical_and(~latent_sequences_end, latent_sequences)
                next_tokens = torch.where(latent_sequences, self.config.latent_id, next_tokens)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            next_embeds = self.get_input_embeddings()(
                torch.where(next_tokens == self.config.latent_id, 0, next_tokens)
            )
            if latent_sequences.any():
                last_layer_hidden_states = outputs.hidden_states[-1]
                new_token_embedding = last_layer_hidden_states[:, -1, :]  # (B, D)
                next_embeds[latent_sequences] = new_token_embedding[latent_sequences]

                if generation_config.latent_do_sample_by == "noise":
                    next_embeds = add_noise(
                        next_embeds,
                        std=generation_config.noise_std,
                        mask=latent_sequences.unsqueeze(-1).expand_as(next_embeds),
                    )

            model_kwargs["inputs_embeds"] = torch.cat(
                [model_kwargs["inputs_embeds"], next_embeds[:, None, :]], dim=1
            )

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            latent_indices = (input_ids == self.config.latent_id).nonzero()[:, 1]
            latent_indices = latent_indices.view(batch_size, generation_config.latent_length)
            # 如果latent_indices每一row都是一样的，则不做cat，直接返回
            if latent_indices.unique(dim=0).shape[0] == 1:
                if generation_config.concat_inputs:
                    latent_thoughts = model_kwargs["inputs_embeds"][:, : latent_indices[0][-1] + 1]
                else:
                    latent_thoughts = model_kwargs["inputs_embeds"][
                        :, latent_indices[0][0] : latent_indices[0][-1] + 1
                    ]
                    assert (
                        input_ids[:, latent_indices[0][0] : latent_indices[0][-1] + 1]
                        == self.config.latent_id
                    ).all()
            else:
                latent_thoughts = []
                lengths = set()
                for b in range(batch_size):
                    first_latent_idx = latent_indices[b][0]
                    last_latent_idx = latent_indices[b][-1]
                    length = last_latent_idx - first_latent_idx + 1
                    lengths.add(length)
                    latent_thoughts.append(
                        model_kwargs["inputs_embeds"][b, first_latent_idx : last_latent_idx + 1]
                    )
                # if they are all the same length, stack, else remain a list
                if len(lengths) == 1:
                    latent_thoughts = torch.stack(latent_thoughts, dim=0)
                else:
                    min_length = min(lengths)
                    latent_thoughts = [
                        latent_thoughts[b][:min_length] for b in range(batch_size)
                    ]
                    latent_thoughts = torch.stack(latent_thoughts, dim=0)
            return LatentGenerateDecoderOnlyOutput(
                sequences=input_ids,
                sequences_embeds=model_kwargs["inputs_embeds"],
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                latent_thoughts=latent_thoughts,
            )
        else:
            return input_ids


    def prepare_inputs_for_generation(
        self,
        # input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(
                past_length, inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
            )

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens

        # 3. Prepare base model inputs
        model_inputs["inputs_embeds"] = inputs_embeds
        assert attention_mask is not None
        do_repeat = attention_mask.shape[0] != inputs_embeds.shape[0]
        if do_repeat:
            num_repeats = inputs_embeds.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask.repeat_interleave(num_repeats, dim=0)
        # 4. Create missing `position_ids` on the fly
        if "position_ids" in set(inspect.signature(self.forward).parameters.keys()):
            if kwargs.get("position_ids") is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                if do_repeat:
                    position_ids = kwargs["position_ids"].repeat_interleave(num_repeats, dim=0)
            kwargs["position_ids"] = position_ids

        if past_key_values is None and kwargs.get("use_cache", True):
            past_key_values = DynamicCache()
        has_kvs = isinstance(past_key_values, tuple) or (
            isinstance(past_key_values, DynamicCache) and past_key_values.seen_tokens > 0
        )
        if has_kvs:
            if do_repeat:
                past_key_values.batch_repeat_interleave(num_repeats)
            model_inputs["inputs_embeds"] = inputs_embeds[:, cache_position]
        model_inputs["past_key_values"] = past_key_values
        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if has_kvs:
                    current_input_length = model_inputs["inputs_embeds"].shape[1]
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )

        model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs
