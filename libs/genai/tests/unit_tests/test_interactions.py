from unittest.mock import AsyncMock, Mock, patch

import pytest

from langchain_google_genai.interactions import GoogleGenAIInteractions

MODEL_NAME = "gemini-2.5-flash"
FAKE_API_KEY = "fake-api-key"


@patch("langchain_google_genai.interactions.Client")
def test_create_interaction(mock_client_class: Mock) -> None:
    mock_client_instance = mock_client_class.return_value

    mock_interaction = Mock()
    mock_interaction.id = "interaction-123"
    mock_client_instance.interactions.create.return_value = mock_interaction

    model = GoogleGenAIInteractions(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
    )

    interaction = model.create(input="Tell me a joke")

    assert interaction.id == "interaction-123"
    mock_client_instance.interactions.create.assert_called_once()


@patch("langchain_google_genai.interactions.Client")
async def test_acreate_interaction(mock_client_class: Mock) -> None:
    mock_client_instance = mock_client_class.return_value

    mock_interaction = Mock()
    mock_interaction.id = "interaction-456"
    mock_client_instance.aio.interactions.create = AsyncMock(return_value=mock_interaction)

    model = GoogleGenAIInteractions(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
    )

    interaction = await model.acreate(input="Tell me another joke")

    assert interaction.id == "interaction-456"
    mock_client_instance.aio.interactions.create.assert_called_once()
