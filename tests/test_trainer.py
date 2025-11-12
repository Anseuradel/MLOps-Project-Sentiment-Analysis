import pytest
import torch
import tempfile
import os
from src.model.trainer import train_epoch, train_model, plot_training_results
from src.model.model import SentimentClassifier
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock

class TestTrainer:
    """Test trainer functionality."""
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader for testing."""
        # Create dummy data that matches our dataset format
        batch_size = 2
        seq_len = 128
        
        # Create multiple batches for testing
        batches = []
        for _ in range(3):  # 3 batches
            batch = {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones((batch_size, seq_len)),
                'labels': torch.randint(0, 5, (batch_size,))
            }
            batches.append(batch)
        
        # Create a simple mock dataloader
        class MockDataLoader:
            def __init__(self, batches):
                self.batches = batches
                self.current = 0
            
            def __iter__(self):
                self.current = 0
                return self
            
            def __next__(self):
                if self.current < len(self.batches):
                    batch = self.batches[self.current]
                    self.current += 1
                    return batch
                raise StopIteration
            
            def __len__(self):
                return len(self.batches)
        
        return MockDataLoader(batches)
    
    def test_train_epoch(self, mock_dataloader):
        """Test training for one epoch."""
        model = SentimentClassifier(n_classes=5)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Mock scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        
        # Mock tqdm to avoid progress bar issues
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr('src.model.trainer.tqdm', lambda x, **kwargs: x)
            
            # Test training epoch
            avg_loss, accuracy = train_epoch(
                model, mock_dataloader, loss_fn, optimizer, scheduler, torch.device('cpu')
            )
        
        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_plot_training_results(self, temp_dir):
        """Test plotting functionality."""
        history = {
            'train_loss': [0.8, 0.6, 0.4],
            'train_acc': [0.6, 0.7, 0.8],
            'val_loss': [0.9, 0.7, 0.5],
            'val_acc': [0.5, 0.6, 0.7]
        }
        
        # Test plotting
        plot_training_results(history, temp_dir)
        
        # Check if plot file was created
        plot_path = os.path.join(temp_dir, "accuracy_and_loss_plot.png")
        assert os.path.exists(plot_path)