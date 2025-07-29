"""Tests for the FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from backend.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-document-qa-system"


@patch('backend.gemini_client.gemini_client.summarize_text')
def test_summarize_endpoint(mock_summarize, client):
    """Test the summarize endpoint."""
    # Mock the Gemini client response
    mock_summarize.return_value = "This is a test summary."
    
    request_data = {
        "text": "This is a long text that needs to be summarized. " * 10,
        "style": "concise",
        "max_length": 50
    }
    
    response = client.post("/summarize", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "summary" in data
    assert "original_length" in data
    assert "summary_length" in data
    assert "compression_ratio" in data


@patch('backend.gemini_client.gemini_client.answer_question')
def test_query_endpoint(mock_answer, client):
    """Test the query endpoint."""
    # Mock the Gemini client response
    mock_answer.return_value = "This is a test answer."
    
    request_data = {
        "question": "What is this document about?",
        "context": "This is a test document about artificial intelligence and machine learning.",
        "include_sources": False
    }
    
    response = client.post("/query", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "This is a test answer."


def test_summarize_validation_error(client):
    """Test summarize endpoint with invalid data."""
    request_data = {
        "text": "",  # Empty text should fail validation
        "style": "concise"
    }
    
    response = client.post("/summarize", json=request_data)
    assert response.status_code == 422  # Validation error


def test_query_validation_error(client):
    """Test query endpoint with invalid data."""
    request_data = {
        "question": "",  # Empty question should fail validation
        "context": "Some context"
    }
    
    response = client.post("/query", json=request_data)
    assert response.status_code == 422  # Validation error
