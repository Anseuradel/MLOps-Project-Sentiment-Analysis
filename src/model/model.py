import torch
import torch.nn as nn
from src import config

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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model. This method defines how data flows through our model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask for padding.

        Returns:
            torch.Tensor: Logits (raw scores before softmax).
        """
        # 1. Pass input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Extract pooled [CLS] representation
        pooled_output = outputs.pooler_output

        # 3. Apply dropout for regularization
        dropped_output = self.dropout(pooled_output)

        # 4. Pass through final linear layer to get logits
        return self.fc(dropped_output)



import torch
import torch.nn as nn
import random

class MockSentimentClassifier(nn.Module):
    """
    Mock model for infrastructure testing.
    No Hugging Face model download required.
    """
    def __init__(self, n_classes=5):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        # Return random logits
        return torch.randn(batch_size, self.n_classes)

    def predict(self, texts):
        # Fake predictions for API testing
        preds = [random.randint(0, self.n_classes - 1) for _ in texts]
        return preds
