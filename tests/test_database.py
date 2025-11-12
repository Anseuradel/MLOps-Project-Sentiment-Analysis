import pytest
import sqlite3
import tempfile
import os
from src.api.database import init_db, insert_prediction, fetch_recent_predictions

class TestDatabase:
    """Test database functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Set environment variable for database path
        os.environ['DB_PATH'] = db_path
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_init_db(self, temp_db):
        """Test database initialization."""
        init_db()
        
        # Verify table was created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='predictions'
        """)
        table_exists = cursor.fetchone() is not None
        
        conn.close()
        assert table_exists
    
    def test_insert_prediction(self, temp_db):
        """Test prediction insertion."""
        init_db()
        
        insert_prediction(
            input_text="Test review",
            predicted_label="positive",
            confidence=0.85,
            model_version="1.0.0",
            model_type="MockSentimentClassifier",
            latency_ms=50.0
        )
        
        # Verify insertion
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_fetch_recent_predictions(self, temp_db):
        """Test fetching recent predictions."""
        init_db()
        
        # Insert multiple predictions
        for i in range(5):
            insert_prediction(
                input_text=f"Review {i}",
                predicted_label="positive",
                confidence=0.8,
                model_version="1.0.0",
                model_type="MockSentimentClassifier",
                latency_ms=50.0
            )
        
        # Fetch recent predictions
        predictions = fetch_recent_predictions(limit=3)
        
        assert len(predictions) == 3
        assert len(predictions[0]) == 7  # 7 columns
