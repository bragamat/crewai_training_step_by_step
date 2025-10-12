#!/usr/bin/env python3
"""
Custom LiteLLM embedding wrapper for CrewAI.
This solves the issue where LiteLLM works but CrewAI's OpenAI provider doesn't.
"""

import os
from typing import List

from chromadb.api.types import EmbeddingFunction
from crewai.rag.embeddings.providers.custom.embedding_callable import (
    CustomEmbeddingFunction,
)
from litellm import embedding as litellm_embedding


class LiteLLMEmbeddingWrapper(CustomEmbeddingFunction, EmbeddingFunction):
    """Custom embedding function that uses LiteLLM for corporate endpoint.

    This class inherits from both CustomEmbeddingFunction (for CrewAI validation)
    and ChromaDB's EmbeddingFunction (for interface compatibility).
    """

    def __init__(self):
        """Initialize with working corporate endpoint configuration."""
        super().__init__()  # Initialize CustomEmbeddingFunction
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
        self.api_base = os.getenv("OPENAI_API_BASE")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Convert input documents to embeddings using LiteLLM.

        Args:
            input: List of documents to embed.

        Returns:
            List of lists of floats representing the embeddings.
        """
        try:
            # Use LiteLLM with our working configuration
            response = litellm_embedding(
                model=self.model, api_base=self.api_base, input=input
            )

            # Extract embeddings from response as List[List[float]]
            embeddings = []
            for item in response["data"]:
                embedding_vector = item["embedding"]
                embeddings.append(embedding_vector)  # Keep as list of floats

            return embeddings

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate embeddings with corporate LiteLLM: {e}"
            )


def get_embedder_config():
    """Get the embedder configuration for CrewAI using corporate endpoint."""
    return {
        "provider": "custom",
        "config": {"embedding_callable": LiteLLMEmbeddingWrapper},
    }
