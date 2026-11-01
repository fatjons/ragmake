from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


class SentenceTransformerEmbedder:
    """Local Hugging Face models via sentence-transformers >= 5. `model` is a model id, or an already-built
    SentenceTransformer-like object (anything with encode_query/encode_document) together with its `model_id`.
    A prompt left as None defers to the prompts the model ships with (EmbeddingGemma, Qwen3)."""

    def __init__(self, model: str | Any, *, model_id: str | None = None, dimensions: int | None = None,
                 query_prompt: str | None = None, document_prompt: str | None = None, batch_size: int = 32):
        if isinstance(model, str):
            from sentence_transformers import SentenceTransformer

            model_id, model = model, SentenceTransformer(model, truncate_dim=dimensions)
        elif model_id is None:
            raise ValueError("model_id is required when passing a model object: it names the embedding cache")
        self.model, self.batch_size = model, batch_size
        self.prompts = {True: query_prompt, False: document_prompt}
        prompts = hashlib.sha256(json.dumps([query_prompt, document_prompt]).encode()).hexdigest()[:8]
        self.name = f"st:{model_id}:{dimensions or 'native'}:{prompts}"

    def embed_texts(self, texts: Sequence[str], query: bool = False) -> list[list[float]]:
        encode = self.model.encode_query if query else self.model.encode_document
        prompt = self.prompts[query]
        return encode(list(texts), **({"prompt": prompt} if prompt else {}), normalize_embeddings=True,
                      batch_size=self.batch_size, convert_to_numpy=True).tolist()
