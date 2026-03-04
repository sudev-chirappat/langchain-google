from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.genai.types import (
    GenerateImagesResponse,
    GeneratedImage,
    Image,
)

from langchain_google_genai.images import GoogleGenAIImageModel

MODEL_NAME = "imagen-3.0-generate-001"
FAKE_API_KEY = "fake-api-key"


@patch("langchain_google_genai.images.Client")
def test_generate_images(mock_client_class: Mock) -> None:
    mock_client_instance = mock_client_class.return_value

    # Mock response
    mock_image = Image(image_bytes=b"fake_image_data")
    mock_gen_image = GeneratedImage(image=mock_image)
    mock_response = GenerateImagesResponse(generated_images=[mock_gen_image])

    mock_client_instance.models.generate_images.return_value = mock_response

    model = GoogleGenAIImageModel(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
    )

    images = model.generate_images(prompt="A beautiful sunset")

    assert len(images) == 1
    assert images[0].image_bytes == b"fake_image_data"
    mock_client_instance.models.generate_images.assert_called_once()


@patch("langchain_google_genai.images.Client")
async def test_agenerate_images(mock_client_class: Mock) -> None:
    mock_client_instance = mock_client_class.return_value

    # Mock async response
    mock_image = Image(image_bytes=b"fake_image_data")
    mock_gen_image = GeneratedImage(image=mock_image)
    mock_response = GenerateImagesResponse(generated_images=[mock_gen_image])

    mock_client_instance.aio.models.generate_images = AsyncMock(return_value=mock_response)

    model = GoogleGenAIImageModel(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
    )

    images = await model.agenerate_images(prompt="A beautiful sunset")

    assert len(images) == 1
    assert images[0].image_bytes == b"fake_image_data"
    mock_client_instance.aio.models.generate_images.assert_called_once()


@patch("langchain_google_genai.images.Client")
def test_upscale_image(mock_client_class: Mock) -> None:
    mock_client_instance = mock_client_class.return_value

    mock_image = Image(image_bytes=b"upscaled_image_data")
    mock_gen_image = GeneratedImage(image=mock_image)
    mock_response = GenerateImagesResponse(generated_images=[mock_gen_image])

    mock_client_instance.models.upscale_image.return_value = mock_response

    model = GoogleGenAIImageModel(
        model="imagen-3.0-upscale-001",
        api_key=FAKE_API_KEY,
    )

    upscaled = model.upscale_image(image=b"original_data")

    assert upscaled.image_bytes == b"upscaled_image_data"
    mock_client_instance.models.upscale_image.assert_called_once()
