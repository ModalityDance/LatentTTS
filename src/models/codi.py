from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.cache_utils import Cache


class CODIGPT2Config(GPT2Config):
    def __init__(
        self,
        latent_id: int = -100,
        latent_start_id: int = -100,
        latent_end_id: int = -100,
        target_id: int = -100,
        projector: bool = False,
        projector_dropout: float = 0.0,
        projector_hidden_size: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_id = latent_id
        self.latent_start_id = latent_start_id
        self.latent_end_id = latent_end_id

        self.projector = projector
        self.projector_dropout = projector_dropout
        self.projector_hidden_size = projector_hidden_size


class CODIGPT2(GPT2LMHeadModel):
    config_class = CODIGPT2Config

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.projector:
            self.projector = nn.Sequential(
                nn.Dropout(config.projector_dropout),
                nn.Linear(config.hidden_size, config.projector_hidden_size),
                nn.GELU(),
                nn.Linear(config.projector_hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
            )
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Tuple[Tuple[torch.Tensor]], Cache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        latent_embeds: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[List[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)  # (batch_size, seq_len, hidden_size)
            # replace latent embeddings with latent_embeds
            if latent_embeds is not None:
                for i, latent_embed in enumerate(latent_embeds):
                    latent_indices = (input_ids[i] == self.config.latent_id).nonzero()
                    _start = latent_indices.min()
                    _end = latent_indices.max() + 1
                    inputs_embeds[i, _start:_end] = latent_embed
            input_ids = None
        result = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if output_hidden_states and self.config.projector:
            assert return_dict, "return_dict must be True if output_hidden_states is True, 要不然老子搞不清楚"
            last_layer_hidden_states = result.hidden_states[-1]
            extended_hidden_states = self.projector(last_layer_hidden_states)
            result.hidden_states = result.hidden_states + (extended_hidden_states,)

        return result
