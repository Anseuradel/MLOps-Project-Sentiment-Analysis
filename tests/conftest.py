import pytest
import pandas as pd
import torch
import tempfile
import os
from unittest.mock import Mock

@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'text': [
            'I love this product!',
            'This is terrible',
            'It is okay, nothing special',
            'Absolutely fantastic!',
            'Worst experience ever'
        ],
        'rating': [5, 1, 3, 5, 1],
        'label_id': [4, 0, 2, 4, 0]
    })

@pytest.fixture
def sample_raw_dataframe():
    """Raw DataFrame with title and text separated."""
    return pd.DataFrame({
        'title': ['Great!', 'Bad', None, 'Amazing'],
        'text': ['I love this', 'I hate this', 'It is okay', 'Fantastic product'],
        'rating': [5, 1, 3, 4]
    })

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = Mock()
    mock.encode_plus.return_value = {
        'input_ids': torch.tensor([[101, 7592, 2023, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1]])
    }
    return mock

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'MAX_LEN': 128,
        'BATCH_SIZE': 2,
        'N_CLASSES': 5,
        'MODEL_NAME': 'prajjwal1/bert-tiny',
        'TOKENIZER_NAME': 'prajjwal1/bert-tiny'
    }
