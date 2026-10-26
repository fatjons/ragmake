from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from ragmake.models import Chunk, Hit

_SELECT = ["id", "corpus_id", "document_id", "chunk_id", "text", "char_start", "char_end", "content_hash", "metadata"]


class AzureBlobArtifactStore:
    def __init__(self, container_client: Any, prefix: str = ""):
        self.container = container_client
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""

    def read_json(self, path: str) -> Any | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return json.loads(self.container.download_blob(self.prefix + path).readall())
        except ResourceNotFoundError:
            return None

    def write_json(self, path: str, data: Any) -> None:
        self.container.upload_blob(self.prefix + path, json.dumps(data), overwrite=True)

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        blobs = self.container.list_blobs(name_starts_with=self.prefix + prefix)
        return [(b.name[len(self.prefix):], json.loads(self.container.download_blob(b.name).readall())) for b in blobs]

    def delete(self, path: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            self.container.delete_blob(self.prefix + path)
        except ResourceNotFoundError:
            pass


class AzureAISearchVectorStore:
    def __init__(self, search_client: Any, index_client: Any | None = None, index_name: str | None = None,
                 vector_field: str = "vector"):
        self.search_client, self.index_client = search_client, index_client
        self.index_name = index_name or search_client._index_name
        self.vector_field = vector_field

    def ensure_index(self, dimensions: int) -> None:
        from azure.core.exceptions import ResourceNotFoundError
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration, SearchableField, SearchField, SearchFieldDataType, SearchIndex,
            SimpleField, VectorSearch, VectorSearchProfile,
        )

        try:
            self.index_client.get_index(self.index_name)
            return
        except ResourceNotFoundError:
            pass
        T = SearchFieldDataType
        fields = [
            SimpleField(name="id", type=T.String, key=True),
            SimpleField(name="corpus_id", type=T.String, filterable=True),
            SimpleField(name="document_id", type=T.String, filterable=True),
            SimpleField(name="chunk_id", type=T.Int32),
            SearchableField(name="text"),
            SimpleField(name="char_start", type=T.Int32),
            SimpleField(name="char_end", type=T.Int32),
            SimpleField(name="content_hash", type=T.String),
            SimpleField(name="metadata", type=T.String),
            SearchField(name=self.vector_field, type=T.Collection(T.Single), searchable=True,
                        vector_search_dimensions=dimensions, vector_search_profile_name="ragmake"),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="ragmake-hnsw")],
            profiles=[VectorSearchProfile(name="ragmake", algorithm_configuration_name="ragmake-hnsw")],
        )
        self.index_client.create_index(SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search))

    def _doc(self, chunk: Chunk, vector: Sequence[float]) -> dict[str, Any]:
        return {
            "id": chunk.key, "corpus_id": chunk.corpus_id, "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id, "text": chunk.text, "char_start": chunk.char_start,
            "char_end": chunk.char_end, "content_hash": chunk.content_hash,
            "metadata": json.dumps(chunk.metadata), self.vector_field: list(vector),
        }

    @staticmethod
    def _chunk(doc: Any) -> Chunk:
        return Chunk(doc["corpus_id"], doc["document_id"], doc["chunk_id"], doc["id"], doc["text"],
                     doc["char_start"], doc["char_end"], doc["content_hash"], json.loads(doc["metadata"]))

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        docs = [self._doc(c, v) for c, v in zip(chunks, vectors, strict=True)]
        for i in range(0, len(docs), 1000):  # service limit per request
            self.search_client.upload_documents(docs[i : i + 1000])

    def delete_chunks(self, keys: Sequence[str]) -> None:
        if keys:
            self.search_client.delete_documents([{"id": k} for k in keys])

    def search(self, corpus_id: str, vector: Sequence[float], top_k: int) -> list[Hit]:
        from azure.search.documents.models import VectorizedQuery

        query = VectorizedQuery(vector=list(vector), k_nearest_neighbors=top_k, fields=self.vector_field)
        results = self.search_client.search(
            search_text=None, vector_queries=[query], top=top_k, select=_SELECT,
            filter="corpus_id eq '{}'".format(corpus_id.replace("'", "''")),
        )
        return [Hit(self._chunk(r), float(r["@search.score"])) for r in results]
