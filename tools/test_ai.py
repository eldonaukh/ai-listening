import pytest
import json
from unittest.mock import MagicMock, patch
from ai import SentimentAnalyzer, OpenAIClient, PoeClient, DeepSeekClient, get_analyzer

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mock_provider():
    """Creates a mock LLM provider."""
    provider = MagicMock()
    return provider

@pytest.fixture
def analyzer(mock_provider):
    """Creates an analyzer instance with the mock provider."""
    return SentimentAnalyzer(mock_provider)

# ==========================================
# Tests for SentimentAnalyzer
# ==========================================

def test_analyze_success(analyzer, mock_provider):
    """Test that valid JSON from the provider is parsed correctly."""
    # Simulate a valid JSON response string from the LLM
    mock_json_response = '{"sentiment": "P", "reason": "Good feedback"}'
    mock_provider.get_completion.return_value = mock_json_response

    result = analyzer.analyze("Some user text")

    assert result["sentiment"] == "P"
    assert result["reason"] == "Good feedback"
    # Verify the provider was called with the system prompt and user text
    mock_provider.get_completion.assert_called_once()
    args = mock_provider.get_completion.call_args[0][0]
    assert args[0]['role'] == 'system'
    assert args[1]['content'] == 'Some user text'

def test_analyze_json_failure(analyzer, mock_provider):
    """Test that invalid JSON returns the error structure."""
    # Simulate the LLM returning plain text instead of JSON
    mock_provider.get_completion.return_value = "I think this is positive."

    result = analyzer.analyze("Some user text")

    assert result["sentiment"] == "Error"
    assert result["reason"] == "JSON Parse Error"

# ==========================================
# Tests for Factory & Clients
# ==========================================

def test_get_analyzer_factory():
    """Test that the factory returns the correct client types."""
    # We patch the clients so we don't actually initialize OpenAI connections
    with patch('ai.PoeClient'), patch('ai.DeepSeekClient'), patch('ai.OpenAIClient'):
        a1 = get_analyzer("poe")
        assert isinstance(a1.provider, PoeClient)

        a2 = get_analyzer("deepseek")
        assert isinstance(a2.provider, DeepSeekClient)

        a3 = get_analyzer("gpt")
        assert isinstance(a3.provider, OpenAIClient)

def test_openai_client_call():
    """Test that the OpenAIClient calls the library correctly."""
    # Mock the actual 'openai.OpenAI' class imported in ai.py
    with patch('ai.OpenAI') as MockOpenAI:
        # Setup the mock client instance inside the wrapper
        mock_api_instance = MockOpenAI.return_value
        mock_api_instance.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Mocked Response"))
        ]

        client = OpenAIClient()
        response = client.get_completion([{"role": "user", "content": "hi"}])

        assert response == "Mocked Response"
        mock_api_instance.chat.completions.create.assert_called_once()