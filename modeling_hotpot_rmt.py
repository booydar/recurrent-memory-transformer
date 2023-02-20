import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

# from ...activations import ACT2FN, gelu
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
# from transformers.modeling_outputs import (
#     SequenceClassifierOutput,
#     TokenClassifierOutput)


@dataclass
class HotpotOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    span_logits: torch.FloatTensor = None
    ans_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ModelForHotpot(torch.nn.Module):

    def __init__(self, base_model, **model_kwargs):
        super().__init__()
        self.model = base_model

        # self.num_labels = 3

        # self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            self.model.config['classifier_dropout'] if self.model.config['classifier_dropout'] is not None else self.model.config['hidden_dropout_prob']
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.span_classifier = nn.Linear(self.model.config['hidden_size'], 3)
        self.ans_classifier = nn.Linear(self.model.config['hidden_size'], 3)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], HotpotOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        span_logits = self.span_classifier(sequence_output)
        ans_logits = self.ans_classifier(sequence_output[:,0,:])

        # loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(ans_logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(ans_logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(ans_logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(ans_logits, labels)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(span_logits.view(-1, 3), labels.view(-1)) + loss_fct(ans_logits.view(-1, 3), span_labels.view(-1))

        if not return_dict:
            output = (ans_logits, span_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return HotpotOutput(
            loss=loss,
            ans_logits=ans_logits,
            span_logits=span_logits
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


