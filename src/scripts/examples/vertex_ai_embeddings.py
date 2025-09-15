#!/usr/bin/env python3
"""
VertexAI Embeddings wrapper for LangChain integration.
"""

import os
from typing import List

from dotenv import load_dotenv
from google import genai
from google.oauth2 import service_account
from langchain.embeddings.base import Embeddings

load_dotenv()


class VertexAIEmbeddings(Embeddings):
    """VertexAI Embeddings wrapper for LangChain."""

    def __init__(self, project_id: str, location: str = "europe-west1", model: str = "text-embedding-005"):
        """
        Initialize VertexAI Embeddings.

        Args:
            project_id: GCP project ID
            location: GCP location/region
            model: Embedding model name
        """
        self.project_id = project_id
        self.location = location
        self.model = model

        # Setup credentials
        scopes = [
            "https://www.googleapis.com/auth/generative-language",
            "https://www.googleapis.com/auth/cloud-platform",
        ]

        credentials = service_account.Credentials.from_service_account_file(
            os.getenv("CREDENTIALS_PATH_EMBEDDINGS"), scopes=scopes
        )

        self.client = genai.Client(vertexai=True, project=project_id, location=location, credentials=credentials)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.models.embed_content(model=self.model, contents=texts)

        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.models.embed_content(model=self.model, contents=[text])

        return response.embeddings[0].values
