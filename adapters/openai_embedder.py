from __future__ import annotations

from typing import Any, Sequence

from ..models import ChunkRecord


class OpenAICompatibleEmbedder:
    """Embedder adapter for OpenAI-style clients.

    Works with either `openai.OpenAI`, `openai.AzureOpenAI`, or any compatible
    client exposing `client.embeddings.create(...)`.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        batch_size: int = 128,
        dimensions: int | None = None,
    ):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions

    @classmethod
    def from_openai(
        cls,
        api_key: str,
        model: str,
        base_url: str | None = None,
        organization: str | None = None,
        batch_size: int = 128,
        dimensions: int | None = None,
    ) -> "OpenAICompatibleEmbedder":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)
        return cls(client=client, model=model, batch_size=batch_size, dimensions=dimensions)

    @classmethod
    def from_azure_openai(
        cls,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        model: str,
        batch_size: int = 128,
        dimensions: int | None = None,
    ) -> "OpenAICompatibleEmbedder":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return cls(client=client, model=model, batch_size=batch_size, dimensions=dimensions)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        vectors: list[list[float]] = []

        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            request: dict[str, Any] = {
                "input": batch,
                "model": self.model,
            }
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**request)
            vectors.extend([list(item.embedding) for item in response.data])

        return vectors


class OpenAISynopsisCompiler:
    """Nectar compiler backed by an OpenAI-style chat completions client.

    Uses chat completions to distill representative corpus chunks into a compact,
    grounded corpus synopsis (nectar). The result is cached and reused until the
    corpus content signature changes.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        max_chunks: int = 40,
        max_chars_per_chunk: int = 320,
        temperature: float = 0.1,
        max_output_tokens: int = 900,
    ):
        self.client = client
        self.model = model
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @classmethod
    def from_openai(
        cls,
        api_key: str,
        model: str,
        base_url: str | None = None,
        organization: str | None = None,
        **kwargs: Any,
    ) -> "OpenAISynopsisCompiler":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)
        return cls(client=client, model=model, **kwargs)

    @classmethod
    def from_azure_openai(
        cls,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        model: str,
        **kwargs: Any,
    ) -> "OpenAISynopsisCompiler":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return cls(client=client, model=model, **kwargs)

    def compile_synopsis(
        self,
        corpus_id: str,
        chunks: Sequence[ChunkRecord],
        previous_synopsis: str | None = None,
    ) -> str:
        excerpt_lines = []
        for chunk in chunks[: self.max_chunks]:
            excerpt = " ".join(chunk.text.split())[: self.max_chars_per_chunk].rstrip()
            excerpt_lines.append(f"[{chunk.document_id}#{chunk.chunk_id}] {excerpt}")

        prior = previous_synopsis or "None"
        user_prompt = (
            f"Corpus id: {corpus_id}\n"
            f"Previous synopsis: {prior}\n\n"
            "Representative chunks:\n"
            f"{chr(10).join(excerpt_lines)}\n\n"
            "Produce a compact corpus synopsis with:\n"
            "- scope and purpose\n"
            "- main entities or concepts\n"
            "- recurring terminology\n"
            "- caveats or notable constraints\n"
            "Stay grounded in the provided chunks."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You compile operational knowledge capsules for stateful RAG systems. "
                        "Summaries must be precise, compact, and traceable to the provided material."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
