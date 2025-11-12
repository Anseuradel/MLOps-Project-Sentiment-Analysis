import pytest
import sqlite3
import tempfile
import os
from src.api.database import init_db, insert_prediction, fetch_recent_predictions
from unittest.mock import patch

class TestDatabaseSimple:
    """Simplified database tests."""
    
    def test_init_db_creates_table(self):
        """Test that init_db creates the predictions table."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Mock DB_PATH to use our temp file
            with patch('src.api.database.DB_PATH', db_path):
                init_db()
                
                # Verify table was created
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
                table_exists = cursor.fetchone() is not None
                conn.close()
                
                assert table_exists
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_insert_and_fetch(self):
        """Test inserting and fetching predictions."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            with patch('src.api.database.DB_PATH', db_path):
                init_db()
                
                # Insert a prediction
                insert_prediction(
                    input_text="Test review",
                    predicted_label="positive",
                    confidence=0.85,
                    model_version="1.0.0",
                    model_type="MockSentimentClassifier",
                    latency_ms=50.0
                )
                
                # Fetch predictions
                predictions = fetch_recent_predictions(limit=1)
                
                assert len(predictions) == 1
                assert predictions[0][2] == "positive"  # predicted_label
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)