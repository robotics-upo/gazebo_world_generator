import pytest
import requests
from unittest.mock import MagicMock, patch
from gazebo_world_generator.src.models.online_database import OnlineModelDatabase

@pytest.fixture
def online_db():
    return OnlineModelDatabase()

def test_make_request_with_retry_success(online_db):
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response
        
        res = online_db._make_request_with_retry("http://test.com")
        assert res.status_code == 200

def test_make_request_with_retry_failure(online_db):
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        res = online_db._make_request_with_retry("http://test.com")
        assert res is None

def test_search_gazebosim_models(online_db):
    with patch.object(online_db, '_make_request_with_retry') as mock_req:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Fuel API structure
        mock_response.json.return_value = [
            {"name": "test_model", "owner": "test_owner", "description": "test"}
        ]
        mock_req.return_value = mock_response
        
        models = online_db._search_gazebosim_models(["desk"])
        assert len(models) == 1
        assert models[0]['name'] == 'test_model'
        assert 'fuel' in models[0]['source']

def test_generate_search_terms_fallback(online_db):
    # Test fallback when LLM is None
    online_db.llm = None
    terms = online_db._generate_search_terms("office desk with lamp")
    assert any("desk" in t for t in terms)
    assert any("lamp" in t for t in terms)
