from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

from google.genai.client import Client
from google.genai.errors import ClientError
from google.genai.types import (
    HttpOptions,
)
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_google_genai._common import _BaseGoogleGenerativeAI, get_user_agent

logger = logging.getLogger(__name__)


class GoogleGenAIInteractions(_BaseGoogleGenerativeAI):
    """Google GenAI Interactions API integration.

    Provides a unified interface for stateful conversations, tool orchestration,
    and specialized agents like Deep Research.
    """

    client: Optional[Client] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and builds client."""
        additional_headers = self.additional_headers or {}
        _, user_agent = get_user_agent("GoogleGenAIInteractions")
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
            if self.model and self.model.startswith("models/"):
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

    def create(
        self,
        input: Union[str, List[Any]],
        model: Optional[str] = None,
        agent: Optional[str] = None,
        previous_interaction_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a new interaction.

        Args:
            input: The input text or multimodal parts.
            model: The model to use (if not using an agent).
            agent: The specialized agent to use (e.g., 'deep-research-pro-preview-12-2025').
            previous_interaction_id: ID of the previous interaction for stateful conversations.
            **kwargs: Additional arguments for the Interactions API.

        Returns:
            An Interaction object.
        """
        if self.client is None:
            raise ValueError("Client not initialized.")

        try:
            return self.client.interactions.create(
                model=model or self.model,
                agent=agent,
                input=input,
                previous_interaction_id=previous_interaction_id,
                **kwargs,
            )
        except ClientError as e:
            raise ValueError(f"Error creating interaction: {e}") from e

    async def acreate(
        self,
        input: Union[str, List[Any]],
        model: Optional[str] = None,
        agent: Optional[str] = None,
        previous_interaction_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronously create a new interaction."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        try:
            return await self.client.aio.interactions.create(
                model=model or self.model,
                agent=agent,
                input=input,
                previous_interaction_id=previous_interaction_id,
                **kwargs,
            )
        except ClientError as e:
            raise ValueError(f"Error creating interaction: {e}") from e

    def get(self, id: str) -> Any:
        """Get an existing interaction by ID."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        try:
            return self.client.interactions.get(id=id)
        except ClientError as e:
            raise ValueError(f"Error getting interaction: {e}") from e

    async def aget(self, id: str) -> Any:
        """Asynchronously get an existing interaction by ID."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        try:
            return await self.client.aio.interactions.get(id=id)
        except ClientError as e:
            raise ValueError(f"Error getting interaction: {e}") from e

    def list(self, **kwargs: Any) -> Any:
        """List interactions."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        try:
            return self.client.interactions.list(**kwargs)
        except ClientError as e:
            raise ValueError(f"Error listing interactions: {e}") from e

    async def alist(self, **kwargs: Any) -> Any:
        """Asynchronously list interactions."""
        if self.client is None:
            raise ValueError("Client not initialized.")

        try:
            return await self.client.aio.interactions.list(**kwargs)
        except ClientError as e:
            raise ValueError(f"Error listing interactions: {e}") from e
