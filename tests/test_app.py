import pytest
from fastapi.testclient import TestClient
from src.api.app import app, model, try_load_real_model
from unittest.mock import Mock, patch

class TestApp:
    """Test FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_predict_endpoint_mock_model(self, client):
        """Test prediction endpoint with mock model."""
        # Mock the global model
        with patch('src.api.app.model') as mock_model:
            mock_model.predict.return_value = [2]  # neutral
            mock_model.predict_proba.return_value = [[0.1, 0.1, 0.6, 0.1, 0.1]]
            mock_model.__class__.__name__ = "MockSentimentClassifier"
            
            response = client.post(
                "/predict",
                json={"text": "This is a test review"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "prediction" in data
            assert "confidence" in data
    
    def test_reload_model_endpoint(self, client):
        """Test model reloading endpoint."""
        response = client.post(
            "/reload_model",
            json={"use_mock": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
