import pytest
import pandas as pd
import tempfile
import os
from src.model.data_extraction import load_file_by_type, load_data, map_rating_to_label

class TestDataExtraction:
  """Test data extraction functionality"""

  def test_map_rating_to_label(self):
        """Test rating to label mapping."""
        assert map_rating_to_label(1) == ("very negative", 0)
        assert map_rating_to_label(3) == ("neutral", 2)
        assert map_rating_to_label(5) == ("very positive", 4)
        assert map_rating_to_label(2.7) == ("neutral", 2)  # Test rounding  

  def test_load_csv_file(self, temp_dir):
        """Test loading CSV files."""
        # Create test CSV
        csv_path = os.path.join(temp_dir, "test.csv")
        test_data = pd.DataFrame({
            'text': ['test1', 'test2'],
            'rating': [1, 5]
        })
        test_data.to_csv(csv_path, index=False)
        
        # Load and verify
        df = load_file_by_type(csv_path)
        assert len(df) == 2
        assert 'text' in df.columns


  def test_load_json_file(self, temp_dir):
        """Test loading JSON files."""
        json_path = os.path.join(temp_dir, "test.json")
        test_data = [
            {'text': 'test1', 'rating': 1},
            {'text': 'test2', 'rating': 5}
        ]
        import json
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        df = load_file_by_type(json_path)
        assert len(df) == 2

  
  def test_load_data_with_rating_text(self, sample_raw_dataframe, temp_dir):
        """Test loading data with rating and text columns."""
        # Save sample data
        csv_path = os.path.join(temp_dir, "test.csv")
        sample_raw_dataframe.to_csv(csv_path, index=False)
        
        # Test loading
        df = load_data(csv_path)
        assert 'text' in df.columns
        assert 'label_id' in df.columns
        assert 'label_text' in df.columns
        assert len(df) > 0

  def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_file_by_type("nonexistent_file.csv")

  def test_unsupported_format(self, temp_dir):
        """Test handling of unsupported file formats."""
        invalid_path = os.path.join(temp_dir, "test.xyz")
        with open(invalid_path, 'w') as f:
            f.write("test")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_file_by_type(invalid_path)

  
