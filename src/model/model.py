import torch
import torch.nn as nn
import config
import numpy as np

from transformers import BertModel

# My final model
class SentimentClassifier(nn.Module):
    """
    Sentiment Classification Model using BERT as a feature extractor.
    This model uses BERT to generate text representations and adds a classification head.
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
            model_name (str): Pretrained BERT model name from Hugging Face.
            dropout_prob (float): Dropout probability for regularization to prevent overfitting.
        """
        super(SentimentClassifier, self).__init__()
        # Load pre-trained BERT model for feature extraction
        self.bert = BertModel.from_pretrained(model_name)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_prob)
        # Final fully connected layer for classification
        # Input size: BERT hidden size (e.g., 768 for bert-base-uncased)
        # Output size: number of sentiment classes
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    # Original forward function (commented out for reference)
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

    # Enhanced forward function to solve "horrible" prediction bias
    def forward(self, input_ids, attention_mask):
        """
        Enhanced forward pass with mean pooling to address prediction bias issues.
        Uses mean pooling across all tokens instead of just the [CLS] token.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs of shape [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask for padding tokens of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits (raw scores before softmax) of shape [batch_size, n_classes]
        """
        # Get BERT outputs - last_hidden_state contains representations for all tokens
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Shape: [batch, seq_len, hidden_size]
        
        # Mean pooling: average token representations while ignoring padding tokens
        # Expand attention mask to match hidden state dimensions [batch, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum the hidden states, but only for non-padding tokens (where mask == 1)
        masked_sum = torch.sum(last_hidden_state * mask, dim=1)  # Sum along sequence length
        
        # Count how many non-padding tokens we have for each sequence
        mask_sum = torch.clamp(mask.sum(dim=1), min=1e-9)  # Avoid division by zero
        
        # Compute mean by dividing sum by count of non-padding tokens
        mean_pooled = masked_sum / mask_sum  # Shape: [batch, hidden_size]
    
        # Apply dropout for regularization
        dropped_output = self.dropout(mean_pooled)
        
        # Final classification layer to get logits
        logits = self.fc(dropped_output)
        return logits


class MockSentimentClassifier:
    """
    Mock classifier for testing and development purposes.
    Provides the same interface as the real classifier but returns random predictions.
    Useful for testing pipelines without running actual model inference.
    """
    
    def __init__(self, n_classes=5):
        """
        Initialize the mock classifier.
        
        Args:
            n_classes (int): Number of output classes
        """
        self.n_classes = n_classes
        # Define mock sentiment labels (note: 6 labels for 5 classes - may need adjustment)
        self.labels = ["Horrible", "very negative", "negative", "neutral", "positive", "very positive"]

    def predict(self, texts):
        """
        Generate random class predictions for input texts.
        
        Args:
            texts (list): List of text strings to classify
            
        Returns:
            list: Random class indices for each text
        """
        return [np.random.randint(0, self.n_classes) for _ in texts]

    def predict_proba(self, texts):
        """
        Generate random probability distributions for input texts.
        
        Args:
            texts (list): List of text strings to classify
            
        Returns:
            numpy.ndarray: Random probability distributions of shape [len(texts), n_classes]
        """
        # Generate random numbers and normalize to get valid probability distributions
        probs = np.random.rand(len(texts), self.n_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize each row to sum to 1
        return probs

    def get_label(self, index):
        """
        Get the label string for a given class index.
        
        Args:
            index (int): Class index
            
        Returns:
            str: Corresponding label string
        """
        return self.labels[index]