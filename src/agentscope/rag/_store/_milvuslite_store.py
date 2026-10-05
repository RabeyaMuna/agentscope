# -*- coding: utf-8 -*-
"""The Milvus Lite vector store implementation."""
from typing import Any, Literal, TYPE_CHECKING

from .._document import DocMetadata
from .._reader import Document
from ._store_base import VDBStoreBase
from ...types import Embedding

if TYPE_CHECKING:
    from pymilvus import MilvusClient
else:
    MilvusClient = "pymilvus.MilvusClient"


class MilvusLiteStore(VDBStoreBase):
    """The Milvus Lite vector store implementation, supporting both local and
    remote Milvus instances.

    .. note:: In Milvus Lite, we use the scalar fields to store the metadata,
    including the document ID, chunk ID, and original content. The new
    MilvusClient API is used for simplified operations.

    """

    def __init__(
        self,
        uri: str = "./milvus_demo.db",
        collection_name: str = "demo_collection",
        dimensions: int = 768,
        distance: Literal["COSINE", "L2", "IP"] = "COSINE",
        token: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        collection_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Milvus Lite vector store."""

        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise ImportError(
                "Milvus client is not installed. Please install it with "
                "`pip install pymilvus`.",
            ) from e

        client_kwargs = client_kwargs or {}
        init_params = {"uri": uri, **client_kwargs}
        if token is not None:
            init_params["token"] = token

        self._client = MilvusClient(**init_params)
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.distance = distance
        self.collection_kwargs = collection_kwargs or {}

    async def _validate_collection(self) -> None:
        """Validate the collection exists, if not, create it."""
        if not self._client.has_collection(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimensions,
                metric_type=self.distance,
                **self.collection_kwargs,
            )

    async def add(self, documents: list[Document], **kwargs: Any) -> None:
        """Add embeddings to the Milvus vector store."""
        await self._validate_collection()

        data = []
        for doc in documents:
            id_str = f"{doc.metadata.doc_id}_{doc.metadata.chunk_id}"
            unique_id = abs(hash(id_str)) % (10**10)
            data.append(
                {
                    "id": unique_id,
                    "vector": doc.embedding,
                    "doc_id": doc.metadata.doc_id,
                    "chunk_id": doc.metadata.chunk_id,
                    "content": str(doc.metadata.content),
                    "total_chunks": doc.metadata.total_chunks,
                },
            )

        self._client.insert(collection_name=self.collection_name, data=data)

    async def search(
        self,
        query_embedding: Embedding,
        limit: int,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search relevant documents from the Milvus vector store."""
        filter_expr = kwargs.get("filter", None)
        output_fields = kwargs.get(
            "output_fields",
            ["doc_id", "chunk_id", "content", "total_chunks"],
        )

        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            filter=filter_expr,
            output_fields=output_fields,
        )

        collected_res = []
        for hits in results:
            for hit in hits:
                if score_threshold is not None and hit["distance"] < score_threshold:
                    continue

                entity = hit["entity"]
                from ...message import TextBlock

                doc_metadata = DocMetadata(
                    content=TextBlock(text=entity.get("content", "")),
                    doc_id=entity.get("doc_id", ""),
                    chunk_id=entity.get("chunk_id", 0),
                    total_chunks=entity.get("total_chunks", 0),
                )
                collected_res.append(
                    Document(
                        embedding=None,
                        score=hit["distance"],
                        metadata=doc_metadata,
                    ),
                )

        return collected_res

    async def delete(
        self,
        ids: list[str] | None = None,
        filter: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete documents from the Milvus vector store."""
        if ids is None and filter is None:
            raise ValueError("Either ids or filter must be provided for deletion.")

        self._client.delete(
            collection_name=self.collection_name,
            ids=ids,
            filter=filter,
        )

    def get_client(self) -> MilvusClient:
        """Get the underlying Milvus client."""
        return self._client
