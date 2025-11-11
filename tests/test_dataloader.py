import pytest
import torch
from src.model.dataloader import SentimentDataset, create_dataloader
from transformers import AutoTokenizer

class TestDataloader:
    """Test dataloader functionality."""
    
    def test_sentiment_dataset(self, mock_tokenizer):
        """Test SentimentDataset class."""
        reviews = ["Great product!", "Terrible experience"]
        labels = [4, 0]
        
        dataset = SentimentDataset(
            reviews=reviews,
            labels=labels,
            tokenizer=mock_tokenizer,
            max_len=128
        )
        
        # Test length
        assert len(dataset) == 2
        
        # Test getitem
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert sample['labels'].dtype == torch.long
    
    def test_create_dataloader(self, sample_dataframe):
        """Test dataloader creation."""
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        dataloader = create_dataloader(
            df=sample_dataframe,
            tokenizer=tokenizer,
            max_len=128,
            batch_size=2
        )
        
        # Test dataloader properties
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert batch['input_ids'].shape[0] == 2  # batch size
