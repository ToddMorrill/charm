import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel

from .utils import get_triples

class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XLMRClassificationPlusTripletLoss(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

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
        anchors: Optional[torch.LongTensor] = None,
        positives: Optional[torch.LongTensor] = None,
        negatives: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        triplet_loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels > 1 and (labels.dtype == torch.long
                                            or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            
            # get the triplet loss if anchors, positives, and negatives are provided
            if anchors is not None and positives is not None and negatives is not None:
                # get the embeddings for the anchor, positive, and negative examples
                # last hidden state has shape num_examples x max_seq_length x hidden_size
                # take <s> token (equiv. to [CLS])
                cls_embeddings = sequence_output[:, 0, :]
                anchor_embeddings = cls_embeddings[anchors]
                positive_embeddings = cls_embeddings[positives]
                negative_embeddings = cls_embeddings[negatives]

                # calculate the triplet loss
                triplet_loss = F.triplet_margin_loss(anchor_embeddings,
                                                    positive_embeddings,
                                                    negative_embeddings,
                                                    margin=1.0,
                                                    p=2.0,
                                                    eps=1e-06,
                                                    swap=False,
                                                    reduction='mean')
        return (loss, logits, triplet_loss)

if __name__ == "__main__":
    model = XLMRClassificationPlusTripletLoss.from_pretrained(
        'xlm-roberta-base',
        num_labels=3,
    )
    # generate random model input and labels from the set {0, 1, 2, 3}
    num_examples = 10
    input_ids = torch.randint(1, 100, (num_examples, 11))
    labels = torch.randint(0, 3, (num_examples, ))

    # generate anchors, positives, and negatives
    anchors, positives, negatives = get_triples(labels, 4)

    # run model
    output = model(input_ids,
                         labels=labels,
                         output_hidden_states=True,
                         anchors=anchors,
                         positives=positives,
                         negatives=negatives)
    print(f'Classification loss: {output[0]}, Triplet loss: {output[2]}')