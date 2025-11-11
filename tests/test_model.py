import pytest
import torch
from src.model.model import SentimentClassifier, MockSentimentClassifier

class TestModel:
    """Test model functionality."""
    
    def test_sentiment_classifier_initialization(self):
        """Test model initialization."""
        model = SentimentClassifier(n_classes=5)
        
        assert model.fc.out_features == 5
        assert hasattr(model, 'bert')
        assert hasattr(model, 'dropout')
    
    def test_sentiment_classifier_forward(self):
        """Test model forward pass."""
        model = SentimentClassifier(n_classes=5)
        
        # Create dummy input
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Check output shape
        assert outputs.shape == (batch_size, 5)  # batch_size x n_classes
    
    def test_mock_classifier(self):
        """Test mock classifier functionality."""
        mock_model = MockSentimentClassifier(n_classes=5)
        
        texts = ["Great!", "Terrible!"]
        
        # Test prediction
        predictions = mock_model.predict(texts)
        assert len(predictions) == 2
        assert all(0 <= pred < 5 for pred in predictions)
        
        # Test probabilities
        probabilities = mock_model.predict_proba(texts)
        assert probabilities.shape == (2, 5)
        assert torch.allclose(torch.tensor(probabilities.sum(axis=1)), torch.tensor([1.0, 1.0]))
        
        # Test label mapping
        label = mock_model.get_label(0)
        assert label == "Horrible"
