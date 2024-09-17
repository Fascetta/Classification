from transformers import AutoModel
import torch
from torch import nn


class NLPClassificationModel(nn.Module):

    def __init__(
        self, model_name: str, num_classes: int, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.encoder = AutoModel.from_pretrained(self.model_name)
        if pretrained == False:
            self.encoder.init_weights()
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, self.num_classes)

    def encode(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask):
        embeddings = self.encode(input_ids, attention_mask)
        logits = self.head(embeddings)
        return logits
