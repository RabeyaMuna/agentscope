# -*- coding: utf-8 -*-
"""Test the RAG store implementations."""
import os
import shutil
from unittest import IsolatedAsyncioTestCase

from agentscope.message import TextBlock
from agentscope.rag import (
    QdrantStore,
    Document,
    DocMetadata,
    MilvusLiteStore,
)


class RAGStoreTest(IsolatedAsyncioTestCase):
    """Test cases for RAG store implementations."""

    async def asyncSetUp(self) -> None:
        """Set up before each test."""
        # Remove the test database path before running tests.
        if os.path.exists("./milvus_demo.db"):
            if os.path.isdir("./milvus_demo.db"):
                shutil.rmtree("./milvus_demo.db")
            else:
                os.remove("./milvus_demo.db")

    async def test_qdrant_store(self) -> None:
        """Test the QdrantStore implementation."""
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            self.skipTest(
                "Skipping QdrantStore test because qdrant-client is not installed. "
                "Install it with `pip install qdrant-client` to enable this test."
            )

        store = QdrantStore(
            location=":memory:",
            collection_name="test",
            dimensions=3,
        )

        await store.add(
            [
                Document(
                    embedding=[0.1, 0.2, 0.3],
                    metadata=DocMetadata(
                        content=TextBlock(
                            type="text",
                            text="This is a test document.",
                        ),
                        doc_id="doc1",
                        chunk_id=0,
                        total_chunks=2,
                    ),
                ),
                Document(
                    embedding=[0.9, 0.1, 0.4],
                    metadata=DocMetadata(
                        content=TextBlock(
                            type="text",
                            text="This is another test document.",
                        ),
                        doc_id="doc1",
                        chunk_id=1,
                        total_chunks=2,
                    ),
                ),
            ],
        )

        res = await store.search(
            query_embedding=[0.15, 0.25, 0.35],
            limit=3,
            score_threshold=0.8,
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(
            res[0].score,
            0.9974149072579597,
        )
        self.assertEqual(
            res[0].metadata.content["text"],
            "This is a test document.",
        )

    async def test_milvus_lite_store(self) -> None:
        """Test the MilvusLiteStore implementation."""
        try:
            import pymilvus  # noqa: F401
            import milvus_lite  # noqa: F401
        except ImportError as e:
            self.skipTest(
                "Skipping MilvusLiteStore test because pymilvus[milvus_lite] is not installed. "
                "Install it with `pip install pymilvus[milvus_lite]` to enable this test."
            )

        store = MilvusLiteStore(
            uri="./milvus_demo.db",
            collection_name="test_milvus",
            dimensions=3,
        )

        await store.add(
            [
                Document(
                    embedding=[0.1, 0.2, 0.3],
                    metadata=DocMetadata(
                        content=TextBlock(
                            type="text",
                            text="This is a test document.",
                        ),
                        doc_id="doc1",
                        chunk_id=0,
                        total_chunks=2,
                    ),
                ),
                Document(
                    embedding=[0.9, 0.1, 0.4],
                    metadata=DocMetadata(
                        content=TextBlock(
                            type="text",
                            text="This is another test document.",
                        ),
                        doc_id="doc1",
                        chunk_id=1,
                        total_chunks=2,
                    ),
                ),
            ],
        )

        res = await store.search(
            query_embedding=[0.15, 0.25, 0.35],
            limit=3,
            score_threshold=0.8,
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(
            res[0].score,
            0.9974148273468018,
        )
        self.assertEqual(
            res[0].metadata.content["text"],
            "This is a test document.",
        )

    async def asyncTearDown(self) -> None:
        """Clean up after tests."""
        # Remove Milvus Lite database path
        if os.path.exists("./milvus_demo.db"):
            if os.path.isdir("./milvus_demo.db"):
                shutil.rmtree("./milvus_demo.db")
            else:
                os.remove("./milvus_demo.db")
