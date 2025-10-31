import torch
import torch.nn as nn
import config

from transformers import BertModel

# My final model
class SentimentClassifier(nn.Module):
    """
    Sentiment Classification Model using BERT as a feature extractor.
    """

    def __init__(
        self,
        n_classes: int,
        model_name: str = config.MODEL_NAME,
        dropout_prob: float = 0.3,
    ):
        """
        Initializes the Sentiment Classifier.

        Args:
            n_classes (int): Number of output classes (e.g., 5 for sentiment classification).
            model_name (str): Pretrained BERT model name.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    # def forward(
    #     self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Forward pass through the model. This method defines how data flows through our model.

    #     Args:
    #         input_ids (torch.Tensor): Tokenized input IDs.
    #         attention_mask (torch.Tensor): Attention mask for padding.

    #     Returns:
    #         torch.Tensor: Logits (raw scores before softmax).
    #     """
    #     # 1. Pass input to BERT
    #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

    #     # 2. Extract pooled [CLS] representation
    #     pooled_output = outputs.pooler_output

    #     # 3. Apply dropout for regularization
    #     dropped_output = self.dropout(pooled_output)

    #     # 4. Pass through final linear layer to get logits
    #     return self.fc(dropped_output)

    # Testing new forward function to solve "horrible"  prediction bias
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Mean pooling (ignoring padding tokens)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_sum = torch.sum(last_hidden_state * mask, dim=1)
        mask_sum = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = masked_sum / mask_sum
    
        dropped_output = self.dropout(mean_pooled)
        logits = self.fc(dropped_output)
        return logits




import numpy as np

class MockSentimentClassifier:
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.labels = ["Horrible", "very negative", "negative", "neutral", "positive", "very positive"]

    def predict(self, texts):
        """Return a random class index for each text."""
        return [np.random.randint(0, self.n_classes) for _ in texts]

    def predict_proba(self, texts):
        """Return random but valid probability distributions."""
        probs = np.random.rand(len(texts), self.n_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def get_label(self, index):
        """Return label string for an index."""
        return self.labels[index]

