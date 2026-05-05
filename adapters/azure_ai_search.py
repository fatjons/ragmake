from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Sequence

from ..models import ChunkRecord, RetrievedChunk


class AzureAISearchVectorStore:
    """Azure AI Search adapter for chunk storage and vector retrieval."""

    def __init__(
        self,
        search_client: Any,
        index_client: Any | None = None,
        index_name: str | None = None,
        vector_field: str = "vector",
    ):
        self.search_client = search_client
        self.index_client = index_client
        self.index_name = index_name or getattr(search_client, "index_name", None) or getattr(search_client, "_index_name", None)
        self.vector_field = vector_field

    @classmethod
    def from_endpoint_and_key(
        cls,
        endpoint: str,
        index_name: str,
        key: str,
    ) -> "AzureAISearchVectorStore":
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        credential = AzureKeyCredential(key)
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        return cls(search_client=search_client, index_client=index_client, index_name=index_name)

    def ensure_index(self, dimensions: int) -> None:
        if self.index_client is None:
            raise RuntimeError("index_client is required to create or update an Azure AI Search index")

        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SearchableField,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        if not self.index_name:
            raise RuntimeError("index_name is required to create or update an Azure AI Search index")
        try:
            self.index_client.get_index(self.index_name)
            return
        except Exception:
            pass

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="corpus_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="chunk_key", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="char_start", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="char_end", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="token_start", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="token_end", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="metadata_json", type=SearchFieldDataType.String),
            SearchField(
                name=self.vector_field,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimensions,
                vector_search_profile_name="stateful-rag-profile",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="stateful-rag-hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="stateful-rag-profile",
                    algorithm_configuration_name="stateful-rag-hnsw",
                )
            ],
        )
        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        self.index_client.create_index(index)

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        docs = []
        for chunk, vector in zip(chunks, vectors):
            docs.append(
                {
                    "id": chunk.chunk_key,
                    "corpus_id": chunk.corpus_id,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_key": chunk.chunk_key,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "content_hash": chunk.content_hash,
                    "metadata_json": json.dumps(chunk.metadata, sort_keys=True),
                    self.vector_field: [float(value) for value in vector],
                }
            )
        if docs:
            self.search_client.upload_documents(docs)

    def delete_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        self.search_client.delete_documents(documents=[{"id": chunk_id} for chunk_id in chunk_ids])

    def list_chunks(self, corpus_id: str, limit: int | None = None) -> list[ChunkRecord]:
        results = self.search_client.search(
            search_text="*",
            filter=f"corpus_id eq '{_escape_filter_value(corpus_id)}'",
            top=limit or 1000,
            select=[
                "corpus_id",
                "document_id",
                "chunk_id",
                "chunk_key",
                "text",
                "char_start",
                "char_end",
                "token_start",
                "token_end",
                "content_hash",
                "metadata_json",
            ],
        )
        return [self._deserialize_chunk(item) for item in results]

    def search(
        self,
        corpus_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=[float(value) for value in query_vector],
            k_nearest_neighbors=top_k,
            fields=self.vector_field,
        )
        results = self.search_client.search(
            search_text=None,
            filter=f"corpus_id eq '{_escape_filter_value(corpus_id)}'",
            vector_queries=[vector_query],
            top=top_k,
            select=[
                "corpus_id",
                "document_id",
                "chunk_id",
                "chunk_key",
                "text",
                "char_start",
                "char_end",
                "token_start",
                "token_end",
                "content_hash",
                "metadata_json",
            ],
        )
        items: list[RetrievedChunk] = []
        for result in results:
            chunk = self._deserialize_chunk(result)
            score = float(result.get("@search.score", 0.0))
            items.append(RetrievedChunk(chunk=chunk, score=score))
        return items

    def _deserialize_chunk(self, item: Any) -> ChunkRecord:
        metadata_json = item.get("metadata_json") or "{}"
        metadata = json.loads(metadata_json)
        chunk = ChunkRecord(
            corpus_id=item["corpus_id"],
            document_id=item["document_id"],
            chunk_id=int(item["chunk_id"]),
            chunk_key=item["chunk_key"],
            text=item["text"],
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            token_start=int(item["token_start"]),
            token_end=int(item["token_end"]),
            content_hash=item["content_hash"],
            metadata=metadata,
        )
        return ChunkRecord(**asdict(chunk))


def _escape_filter_value(value: str) -> str:
    return value.replace("'", "''")
