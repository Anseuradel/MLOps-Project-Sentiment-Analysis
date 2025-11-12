import pytest
import torch
import tempfile
import os
from src.model.trainer import train_epoch, train_model, plot_training_results
from src.model.model import SentimentClassifier
from torch.utils.data import DataLoader, TensorDataset

class TestTrainer:
    """Test trainer functionality."""
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader for testing."""
        # Create dummy data
        input_ids = torch.randint(0, 1000, (10, 128))
        attention_mask = torch.ones((10, 128))
        labels = torch.randint(0, 5, (10,))
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        return DataLoader(dataset, batch_size=2)
    
    def test_train_epoch(self, mock_dataloader):
        """Test training for one epoch."""
        model = SentimentClassifier(n_classes=5)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Mock scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        
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
