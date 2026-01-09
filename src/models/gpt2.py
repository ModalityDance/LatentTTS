from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Config, GPT2Model
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.cache_utils import Cache

from src.models.coconut import COCONUTGPT2Config
from src.models.codi import CODIGPT2Config


class COCONUTGPT2ForTokenClassification(GPT2PreTrainedModel):
    config_class = COCONUTGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(config.hidden_size, 1)

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
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(
                torch.where(input_ids == self.config.latent_id, 0, input_ids)
            )  # (batch_size, seq_len, hidden_size)
            # replace latent embeddings with latent_embeds
            i = 0
            for latent_embed in latent_embeds:
                if not latent_embed.dim() == 2:
                    for j in range(latent_embed.shape[0]):
                        latent_indices = (input_ids[i] == self.config.latent_id).nonzero()
                        _start = latent_indices.min()
                        _end = latent_indices.max() + 1
                        inputs_embeds[i, _start:_end] = latent_embed[j]
                        i += 1
                else:
                    latent_indices = (input_ids[i] == self.config.latent_id).nonzero()
                    _start = latent_indices.min()
                    _end = latent_indices.max() + 1
                    inputs_embeds[i, _start:_end] = latent_embed
                    i += 1

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = transformer_outputs.last_hidden_state
        last_hidden_states = self.dropout(last_hidden_states)
        logits = self.classifier(last_hidden_states)

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return output

        return TokenClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class CODIGPT2ForTokenClassification(GPT2PreTrainedModel):
    config_class = CODIGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(config.hidden_size, 1)

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
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(
                torch.where(input_ids == self.config.latent_id, 0, input_ids)
            )  # (batch_size, seq_len, hidden_size)
            # replace latent embeddings with latent_embeds
            i = 0
            for latent_embed in latent_embeds:
                if not latent_embed.dim() == 2:
                    for j in range(latent_embed.shape[0]):
                        latent_indices = (input_ids[i] == self.config.latent_id).nonzero()
                        _start = latent_indices.min()
                        _end = latent_indices.max() + 1
                        inputs_embeds[i, _start:_end] = latent_embed[j]
                        i += 1
                else:
                    latent_indices = (input_ids[i] == self.config.latent_id).nonzero()
                    _start = latent_indices.min()
                    _end = latent_indices.max() + 1
                    inputs_embeds[i, _start:_end] = latent_embed
                    i += 1

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.config.projector:
            extra_hidden_state = self.projector(transformer_outputs.last_hidden_state)
            if output_hidden_states:
                transformer_outputs.hidden_states = transformer_outputs.hidden_states + (extra_hidden_state,)
            transformer_outputs.last_hidden_state = extra_hidden_state

        last_hidden_states = transformer_outputs.last_hidden_state

        last_hidden_states = self.dropout(last_hidden_states)
        logits = self.classifier(last_hidden_states)

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return output

        return TokenClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
