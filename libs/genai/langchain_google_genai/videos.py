from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, List, Optional, Union

from google.genai.client import Client
from google.genai.errors import ClientError
from google.genai.types import (
    GenerateVideosConfig,
    HttpOptions,
    Image,
    Video,
)
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_google_genai._common import _BaseGoogleGenerativeAI, get_user_agent

logger = logging.getLogger(__name__)


class GoogleGenAIVideoModel(_BaseGoogleGenerativeAI):
    """Google GenAI Video model integration.

    Supports Veo models for video generation.
    """

    client: Optional[Client] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and builds client."""
        additional_headers = self.additional_headers or {}
        _, user_agent = get_user_agent("GoogleGenAIVideoModel")
        headers = {"User-Agent": user_agent, **additional_headers}

        google_api_key = None
        if not self.credentials:
            if isinstance(self.google_api_key, SecretStr):
                google_api_key = self.google_api_key.get_secret_value()
            else:
                google_api_key = self.google_api_key

        http_options = HttpOptions(
            headers=headers,
            client_args=self.client_args,
        )

        if self._use_vertexai:  # type: ignore[attr-defined]
            if self.model.startswith("models/"):
                object.__setattr__(self, "model", self.model.replace("models/", "", 1))

            self.client = Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                credentials=self.credentials,
                http_options=http_options,
            )
        else:
            if not google_api_key:
                msg = "API key required for Gemini Developer API."
                raise ValueError(msg)
            self.client = Client(api_key=google_api_key, http_options=http_options)
        return self

    def generate_videos(
        self,
        prompt: Optional[str] = None,
        image: Optional[Union[Image, str]] = None,
        video: Optional[Union[Video, str]] = None,
        wait_for_completion: bool = True,
        polling_interval: int = 20,
        **kwargs: Any,
    ) -> Union[Any, List[Video]]:
        """Generate videos from text, image, or video input.

        Args:
            prompt: Optional text prompt.
            image: Optional input image (Image object or path/URI).
            video: Optional input video (Video object or path/URI).
            wait_for_completion: Whether to wait for the long-running operation.
            polling_interval: Seconds between polls when waiting.
            **kwargs: Additional arguments for GenerateVideosConfig.

        Returns:
            If wait_for_completion is True, returns a list of generated Video objects.
            Otherwise, returns the long-running operation object.
        """
        if self.client is None:
            raise ValueError("Client not initialized.")

        # Handle file paths/URIs if strings are provided
        input_image = image
        if isinstance(image, str):
            if image.startswith("gs://") or image.startswith("http"):
                input_image = Image(uri=image)
            else:
                input_image = Image.from_file(image)

        input_video = video
        if isinstance(video, str):
            if video.startswith("gs://") or video.startswith("http"):
                input_video = Video(uri=video)
            else:
                input_video = Video.from_file(video)

        config = GenerateVideosConfig(**kwargs)
        try:
            operation = self.client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                image=input_image,
                video=input_video,
                config=config,
            )

            if not wait_for_completion:
                return operation

            while not operation.done:
                time.sleep(polling_interval)
                operation = self.client.operations.get(operation)

            if operation.error:
                raise ValueError(f"Video generation failed: {operation.error}")

            return [gen_vid.video for gen_vid in operation.response.generated_videos]
        except ClientError as e:
            raise ValueError(f"Error starting video generation: {e}") from e

    async def agenerate_videos(
        self,
        prompt: Optional[str] = None,
        image: Optional[Union[Image, str]] = None,
        video: Optional[Union[Video, str]] = None,
        wait_for_completion: bool = True,
        polling_interval: int = 20,
        **kwargs: Any,
    ) -> Union[Any, List[Video]]:
        """Asynchronously generate videos."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        # Handle file paths/URIs if strings are provided
        input_image = image
        if isinstance(image, str):
            if image.startswith("gs://") or image.startswith("http"):
                input_image = Image(uri=image)
            else:
                input_image = Image.from_file(image)

        input_video = video
        if isinstance(video, str):
            if video.startswith("gs://") or video.startswith("http"):
                input_video = Video(uri=video)
            else:
                input_video = Video.from_file(video)

        config = GenerateVideosConfig(**kwargs)
        try:
            operation = await self.client.aio.models.generate_videos(
                model=self.model,
                prompt=prompt,
                image=input_image,
                video=input_video,
                config=config,
            )

            if not wait_for_completion:
                return operation

            while not operation.done:
                await asyncio.sleep(polling_interval)
                operation = await self.client.aio.operations.get(operation)

            if operation.error:
                raise ValueError(f"Video generation failed: {operation.error}")

            return [gen_vid.video for gen_vid in operation.response.generated_videos]
        except ClientError as e:
            raise ValueError(f"Error starting video generation: {e}") from e
