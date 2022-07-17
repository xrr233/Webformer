import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertConfig, BertForMaskedLM

class FraBert(BertForMaskedLM):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self,config,num_type=3):
        super().__init__(config)
        self.type_embedding = nn.Embedding(num_type,self.config.hidden_size)
        self.apply(self._init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_type_idx=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        type_embeddings = self.type_embedding(inputs_type_idx)# batch x n_word_nodes x hidden_size
        inputs_embeds = word_embeddings+type_embeddings

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]  # batch x seq_len x hidden_size
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        word_logits = self.cls(sequence_output)
        word_predict = torch.argmax(word_logits, dim=-1)

        masked_lm_loss = loss_fct(word_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {'loss': masked_lm_loss,
                'word_pred': word_predict,
                'output':sequence_output
                } 