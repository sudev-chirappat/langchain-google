from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.genai.types import (
    GenerateVideosResponse,
    GeneratedVideo,
    Video,
)

from langchain_google_genai.videos import GoogleGenAIVideoModel

MODEL_NAME = "veo-2.0-generate-001"
FAKE_API_KEY = "fake-api-key"


@patch("langchain_google_genai.videos.Client")
def test_generate_videos(mock_client_class: Mock) -> None:
    mock_client_instance = mock_client_class.return_value

    # Mock response
    mock_video = Video(uri="gs://fake/video.mp4")
    mock_gen_video = GeneratedVideo(video=mock_video)

    mock_operation = Mock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = GenerateVideosResponse(generated_videos=[mock_gen_video])

    mock_client_instance.models.generate_videos.return_value = mock_operation

    model = GoogleGenAIVideoModel(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
    )

    videos = model.generate_videos(prompt="A futuristic city")

    assert len(videos) == 1
    assert videos[0].uri == "gs://fake/video.mp4"
    mock_client_instance.models.generate_videos.assert_called_once()


@patch("langchain_google_genai.videos.Client")
async def test_agenerate_videos(mock_client_class: Mock) -> None:
    mock_client_instance = mock_client_class.return_value

    # Mock async response
    mock_video = Video(uri="gs://fake/video.mp4")
    mock_gen_video = GeneratedVideo(video=mock_video)

    mock_operation = Mock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = GenerateVideosResponse(generated_videos=[mock_gen_video])

    mock_client_instance.aio.models.generate_videos = AsyncMock(return_value=mock_operation)

    model = GoogleGenAIVideoModel(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
    )

    videos = await model.agenerate_videos(prompt="A futuristic city")

    assert len(videos) == 1
    assert videos[0].uri == "gs://fake/video.mp4"
    mock_client_instance.aio.models.generate_videos.assert_called_once()
