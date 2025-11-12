import pytest
import pandas as pd
from src.model.data_processing import clean_text, tokenize_texts, preprocess_data
from transformers import AutoTokenizer

class TestDataProcessing:
    """Test data processing functionality."""
    
    def test_clean_text(self):
        """Test text cleaning function."""
        # Test URLs removal
        text_with_url = "Check this out: https://example.com"
        cleaned = clean_text(text_with_url)
        assert "https://example.com" not in cleaned
        
        # Test punctuation removal
        text_with_punct = "Hello, world! This is a test."
        cleaned = clean_text(text_with_punct)
        assert "," not in cleaned
        assert "!" not in cleaned
        
        # Test lowercase conversion
        mixed_case = "Hello WORLD"
        cleaned = clean_text(mixed_case)
        assert cleaned == "hello world"
        
        # Test emoji removal
        text_with_emoji = "I love this! 😍"
        cleaned = clean_text(text_with_emoji)
        assert "😍" not in cleaned
    
    def test_tokenize_texts(self):
        """Test text tokenization."""
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        texts = ["This is a test", "Another test sentence"]
        
        tokenized = tokenize_texts(texts, max_length=128)
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert len(tokenized['input_ids']) == 2
        assert len(tokenized['attention_mask']) == 2
    
    def test_preprocess_data(self, sample_dataframe):
        """Test complete preprocessing pipeline."""
        train_df, val_df = preprocess_data(
            sample_dataframe, test_size=0.2, max_length=128
        )
        
        # Check that dataframes have required columns
        assert 'input_ids' in train_df.columns
        assert 'attention_mask' in train_df.columns
        assert 'text' in train_df.columns
        
        # Check split sizes
        total_samples = len(sample_dataframe)
        expected_val_size = int(total_samples * 0.2)
        assert len(val_df) == expected_val_size
