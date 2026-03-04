from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

from google.genai.client import Client
from google.genai.errors import ClientError
from google.genai.types import (
    EditImageConfig,
    GenerateImagesConfig,
    HttpOptions,
    Image,
    MaskReferenceImage,
    RawReferenceImage,
    UpscaleImageConfig,
)
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_google_genai._common import _BaseGoogleGenerativeAI, get_user_agent

logger = logging.getLogger(__name__)


class GoogleGenAIImageModel(_BaseGoogleGenerativeAI):
    """Google GenAI Image model integration.

    Supports Imagen models for image generation, upscaling, and editing.
    """

    client: Optional[Client] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and builds client."""
        additional_headers = self.additional_headers or {}
        _, user_agent = get_user_agent("GoogleGenAIImageModel")
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

    def generate_images(
        self,
        prompt: str,
        number_of_images: int = 1,
        **kwargs: Any,
    ) -> List[Image]:
        """Generate images from a text prompt.

        Args:
            prompt: The text prompt to generate images from.
            number_of_images: Number of images to generate.
            **kwargs: Additional arguments for GenerateImagesConfig.

        Returns:
            A list of generated Image objects.
        """
        if self.client is None:
            raise ValueError("Client not initialized.")

        config = GenerateImagesConfig(number_of_images=number_of_images, **kwargs)
        try:
            response = self.client.models.generate_images(
                model=self.model,
                prompt=prompt,
                config=config,
            )
            return [gen_img.image for gen_img in response.generated_images]
        except ClientError as e:
            raise ValueError(f"Error generating images: {e}") from e

    async def agenerate_images(
        self,
        prompt: str,
        number_of_images: int = 1,
        **kwargs: Any,
    ) -> List[Image]:
        """Asynchronously generate images from a text prompt."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        config = GenerateImagesConfig(number_of_images=number_of_images, **kwargs)
        try:
            response = await self.client.aio.models.generate_images(
                model=self.model,
                prompt=prompt,
                config=config,
            )
            return [gen_img.image for gen_img in response.generated_images]
        except ClientError as e:
            raise ValueError(f"Error generating images: {e}") from e

    def upscale_image(
        self,
        image: Union[Image, bytes],
        upscale_factor: str = "x2",
        **kwargs: Any,
    ) -> Image:
        """Upscale an existing image.

        Args:
            image: The image to upscale (Image object or bytes).
            upscale_factor: Upscale factor ('x2' or 'x4').
            **kwargs: Additional arguments for UpscaleImageConfig.

        Returns:
            The upscaled Image object.
        """
        if self.client is None:
            raise ValueError("Client not initialized.")

        if isinstance(image, bytes):
            image = Image(image_bytes=image)

        config = UpscaleImageConfig(**kwargs)
        try:
            response = self.client.models.upscale_image(
                model=self.model,
                image=image,
                upscale_factor=upscale_factor,
                config=config,
            )
            return response.generated_images[0].image
        except ClientError as e:
            raise ValueError(f"Error upscaling image: {e}") from e

    async def aupscale_image(
        self,
        image: Union[Image, bytes],
        upscale_factor: str = "x2",
        **kwargs: Any,
    ) -> Image:
        """Asynchronously upscale an existing image."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        if isinstance(image, bytes):
            image = Image(image_bytes=image)

        config = UpscaleImageConfig(**kwargs)
        try:
            response = await self.client.aio.models.upscale_image(
                model=self.model,
                image=image,
                upscale_factor=upscale_factor,
                config=config,
            )
            return response.generated_images[0].image
        except ClientError as e:
            raise ValueError(f"Error upscaling image: {e}") from e

    def edit_image(
        self,
        prompt: str,
        reference_images: List[Union[RawReferenceImage, MaskReferenceImage]],
        **kwargs: Any,
    ) -> List[Image]:
        """Edit an image using prompts and reference images.

        Args:
            prompt: Text prompt describing the edits.
            reference_images: List of reference images (raw or masks).
            **kwargs: Additional arguments for EditImageConfig.

        Returns:
            A list of edited Image objects.
        """
        if self.client is None:
            raise ValueError("Client not initialized.")

        config = EditImageConfig(**kwargs)
        try:
            response = self.client.models.edit_image(
                model=self.model,
                prompt=prompt,
                reference_images=reference_images,
                config=config,
            )
            return [gen_img.image for gen_img in response.generated_images]
        except ClientError as e:
            raise ValueError(f"Error editing image: {e}") from e

    async def aedit_image(
        self,
        prompt: str,
        reference_images: List[Union[RawReferenceImage, MaskReferenceImage]],
        **kwargs: Any,
    ) -> List[Image]:
        """Asynchronously edit an image."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        config = EditImageConfig(**kwargs)
        try:
            response = await self.client.aio.models.edit_image(
                model=self.model,
                prompt=prompt,
                reference_images=reference_images,
                config=config,
            )
            return [gen_img.image for gen_img in response.generated_images]
        except ClientError as e:
            raise ValueError(f"Error editing image: {e}") from e
