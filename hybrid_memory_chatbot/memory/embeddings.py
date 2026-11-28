"""Gemini embedding wrapper for long-term memories.

Self-contained adaptation of the embedding client used by document-chatbot.
Produces 768-dimensional vectors with ``gemini-embedding-2`` and exposes the
standard LangChain ``Embeddings`` interface.
"""

import logging
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load the package .env before config.py builds Settings.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_PACKAGE_DIR / ".env", override=True)

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from config import settings

logger = logging.getLogger(__name__)

# gemini-embedding-2 does not support a task_type config field, so retrieval
# instructions are included in the text itself.
DOCUMENT_PROMPT_PREFIX = "title: none | text: "
QUERY_PROMPT_PREFIX = "task: question answering | query: "


class GeminiEmbedding(Embeddings):
    """LangChain-compatible embeddings backed by the Gemini Embedding API."""

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        max_retries: int | None = None,
        initial_backoff_seconds: float | None = None,
    ) -> None:
        resolved_key = api_key or settings.gemini_api_key

        if not resolved_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        api_key_value = (
            resolved_key.get_secret_value()
            if isinstance(resolved_key, SecretStr)
            else str(resolved_key)
        )

        self.model = model or settings.gemini_embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize = (
            settings.normalize_embeddings if normalize is None else normalize
        )
        self.max_retries = (
            settings.embedding_max_retries
            if max_retries is None
            else max_retries
        )
        self.initial_backoff_seconds = (
            settings.embedding_initial_backoff_seconds
            if initial_backoff_seconds is None
            else initial_backoff_seconds
        )

        self._client = genai.Client(api_key=api_key_value)

        logger.info(
            "GeminiEmbedding initialized: model=%s dimensions=%d batch_size=%d normalize=%s",
            self.model,
            self.dimensions,
            self.batch_size,
            self.normalize,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents, preserving order and batching sequentially."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_embeddings = self._embed_batch(batch, DOCUMENT_PROMPT_PREFIX)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with the retrieval query prefix."""
        embeddings = self._embed_batch([text], QUERY_PROMPT_PREFIX)
        return embeddings[0]

    def _embed_batch(
        self,
        texts: list[str],
        prompt_prefix: str,
    ) -> list[list[float]]:
        """Send one batch to Gemini with retries and exponential backoff."""
        contents = [
            types.Content(parts=[types.Part(text=f"{prompt_prefix}{text}")])
            for text in texts
        ]

        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimensions,
                    ),
                )
                return self._parse_result(result, len(texts))

            except Exception as exc:
                last_exception = exc

                if attempt < self.max_retries and self._is_retryable(exc):
                    backoff = self.initial_backoff_seconds * (2**attempt)
                    logger.warning(
                        "Embedding attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        backoff,
                        self._safe_error_message(exc),
                    )
                    time.sleep(backoff)
                    continue

                logger.error(
                    "Embedding failed after %d attempts: %s",
                    attempt + 1,
                    self._safe_error_message(exc),
                )
                raise

        raise last_exception or RuntimeError("Embedding failed without an exception.")

    def _parse_result(
        self,
        result: types.EmbedContentResponse,
        expected_count: int,
    ) -> list[list[float]]:
        """Validate that every input received exactly one expected vector."""
        embeddings = getattr(result, "embeddings", None)

        if embeddings is None:
            raise ValueError("Gemini embedding response contained no embeddings.")

        if not isinstance(embeddings, (list, tuple)):
            raise ValueError("Gemini embedding response 'embeddings' is not a list.")

        if len(embeddings) != expected_count:
            raise ValueError(
                f"Expected {expected_count} embeddings, got {len(embeddings)}."
            )

        parsed: list[list[float]] = []

        for index, embedding in enumerate(embeddings):
            values = getattr(embedding, "values", None)

            if values is None:
                raise ValueError(f"Embedding {index} has no 'values' field.")

            if len(values) != self.dimensions:
                raise ValueError(
                    f"Embedding {index} has {len(values)} dimensions, "
                    f"expected {self.dimensions}."
                )

            vector = [float(value) for value in values]
            parsed.append(self._normalize(vector) if self.normalize else vector)

        return parsed

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """L2-normalize a vector."""
        norm = sum(value * value for value in vector) ** 0.5

        if norm == 0:
            return vector

        return [value / norm for value in vector]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True for transient API failures."""
        status_code = getattr(exc, "code", None) or getattr(
            exc, "status_code", None
        )

        if status_code in (429, 500, 502, 503, 504):
            return True

        message = str(exc).lower()
        retryable_markers = (
            "rate limit",
            "quota",
            "unavailable",
            "timeout",
            "internal",
        )
        return any(marker in message for marker in retryable_markers)

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        """Log errors without exposing keys or document text."""
        message = str(exc)
        return f"{type(exc).__name__}: {message[:200]}"


@lru_cache(maxsize=1)
def get_embedding_model() -> GeminiEmbedding:
    """Return the process-local embedding client."""
    return GeminiEmbedding()
