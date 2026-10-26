from __future__ import annotations

import json
from typing import Any, Iterable


class AzureBlobArtifactStore:
    """Blob-backed artifact store for ledgers, caches, synopses, and sessions."""

    def __init__(self, container_client: Any, prefix: str = ""):
        self.container_client = container_client
        self.prefix = prefix.strip("/")

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        container_name: str,
        prefix: str = "",
    ) -> "AzureBlobArtifactStore":
        from azure.storage.blob import BlobServiceClient

        service = BlobServiceClient.from_connection_string(connection_string)
        container_client = service.get_container_client(container_name)
        return cls(container_client=container_client, prefix=prefix)

    def read_json(self, path: str) -> Any | None:
        blob_name = self._blob_name(path)
        try:
            data = self.container_client.get_blob_client(blob_name).download_blob().readall()
        except Exception:
            return None
        return json.loads(data.decode("utf-8"))

    def write_json(self, path: str, data: Any) -> None:
        blob_name = self._blob_name(path)
        payload = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
        self.container_client.get_blob_client(blob_name).upload_blob(payload, overwrite=True)

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        blob_prefix = self._blob_name(prefix).rstrip("/")
        items: list[tuple[str, Any]] = []
        for blob in self.container_client.list_blobs(name_starts_with=blob_prefix):
            if not blob.name.endswith(".json"):
                continue
            data = self.container_client.get_blob_client(blob.name).download_blob().readall()
            relative_name = blob.name
            if self.prefix:
                relative_name = blob.name[len(self.prefix) + 1 :]
            items.append((relative_name, json.loads(data.decode("utf-8"))))
        return items

    def delete(self, path: str) -> None:
        blob_name = self._blob_name(path)
        try:
            self.container_client.delete_blob(blob_name)
        except Exception:
            pass

    def _blob_name(self, path: str) -> str:
        normalized = path.strip("/")
        if not self.prefix:
            return normalized
        if not normalized:
            return self.prefix
        return f"{self.prefix}/{normalized}"
