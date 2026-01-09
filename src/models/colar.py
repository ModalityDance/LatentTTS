from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM, KwargsForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


"""
"[PAD][PAD]\
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. \
How much in dollars does she make every day at the farmers' market? \
Let's think step by step:(Thinking speed: 5)###"
"""


class LatentHead(nn.Module):
    def __init__(self, feature_size, intermediate_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )

        self.mean = nn.Linear(intermediate_size, feature_size)

    def forward(self, x):
        x = self.fc(x)
        return self.mean(x)




class ColarLlamaConfig(LlamaConfig):
    def __init__(
        self,
        latent_id: int = -1,
        latent_start_id: int = -1,
        latent_end_id: int = -1,
        latent_intermediate_size: int = 2048,
        latent_embedding_std: float = 0.018,
        **kwargs,
    ):
        super().__init__(**kwargs)
        """
        MODEL_EMB_STD = {
            "DeepSeek-R1-Distill-Qwen-1.5B": 0.03,
            "Llama-3.2-1B-Instruct": 0.018,
            "Llama-3.2-3B-Instruct": 0.018,
            "Llama-3.1-8B-Instruct": 0.008,
            "gpt2": 0.12,
        }
        """
        self.latent_id = latent_id
        self.latent_start_id = latent_start_id
        self.latent_end_id = latent_end_id
        self.latent_intermediate_size = latent_intermediate_size
        self.latent_embedding_std = latent_embedding_std


class ColarLlama(LlamaForCausalLM):
    config_class = ColarLlamaConfig

    def __init__(
        self,
        config,
    ):
        super(LlamaForCausalLM, self).__init__(config)  # this must be called before save hparams
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.lt_head = LatentHead(
            feature_size=self.config.hidden_size,
            intermediate_size=self.config.latent_intermediate_size,
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        latent_embeds: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        assert inputs_embeds is not None, "inputs_embeds must be provided"
        result = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if output_hidden_states:
            last_layer_hidden_states = result.hidden_states[-1]
            extended_hidden_states = self.lt_head(last_layer_hidden_states)
            extended_hidden_states = extended_hidden_states * self.config.latent_embedding_std
            result.hidden_states = result.hidden_states + (extended_hidden_states,)
        return result
