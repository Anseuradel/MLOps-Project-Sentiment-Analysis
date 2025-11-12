import pytest
import pandas as pd
import tempfile
import os
import sys

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.data_extraction import load_file_by_type, load_data, map_rating_to_label

class TestDataExtraction:
    """Test data extraction functionality."""
    
    def test_map_rating_to_label(self):
        """Test rating to label mapping."""
        assert map_rating_to_label(1) == ("very negative", 0)
        assert map_rating_to_label(3) == ("neutral", 2)
        assert map_rating_to_label(5) == ("very positive", 4)
    
    def test_load_csv_file(self, tmp_path):
        """Test loading CSV files."""
        # Create test CSV
        csv_path = tmp_path / "test.csv"
        test_data = pd.DataFrame({
            'text': ['test1', 'test2'],
            'rating': [1, 5]
        })
        test_data.to_csv(csv_path, index=False)
        
        # Load and verify
        df = load_file_by_type(str(csv_path))
        assert len(df) == 2
        assert 'text' in df.columns
    
    def test_load_json_file(self, tmp_path):
        """Test loading JSON files."""
        json_path = tmp_path / "test.json"
        test_data = [
            {'text': 'test1', 'rating': 1},
            {'text': 'test2', 'rating': 5}
        ]
        import json
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        df = load_file_by_type(str(json_path))
        assert len(df) == 2
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_file_by_type("nonexistent_file.csv")
