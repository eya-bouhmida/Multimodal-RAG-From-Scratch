import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Curated list of image filenames worth showing (charts/graphs/medically meaningful
# illustrations) — most extracted images are logos, covers or icons and are excluded.
# See data/processed/image_whitelist.json.
IMAGE_WHITELIST_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "image_whitelist.json"


def _load_image_whitelist() -> set[str] | None:
    if not IMAGE_WHITELIST_PATH.exists():
        return None
    with open(IMAGE_WHITELIST_PATH, encoding="utf-8") as f:
        return set(json.load(f))


class HybridRetriever:
    """Dense (Qdrant) + BM25 hybrid search with RRF fusion and cross-encoder reranking."""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "medlens",
        image_collection_name: str = "medlens_images",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.collection_name = collection_name
        self.image_collection_name = image_collection_name

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        self.image_whitelist = _load_image_whitelist()

        self._chunks = []
        self._bm25 = None
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Scroll the full collection (not just a page) to build the BM25 corpus."""
        self._chunks = []
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            self._chunks.extend(results)
            if offset is None:
                break

        tokenized = [chunk.payload["text"].lower().split() for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    @property
    def text_chunks_count(self) -> int:
        return len(self._chunks)

    @property
    def image_chunks_count(self) -> int:
        return self.client.count(self.image_collection_name).count

    def dense_search(self, query: str, top_k: int = 20):
        query_vector = self.embedding_model.encode(query).tolist()
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

    def bm25_search(self, query: str, top_k: int = 20):
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": self._chunks[idx].id, "score": scores[idx], "payload": self._chunks[idx].payload}
            for idx in top_indices
        ]

    @staticmethod
    def reciprocal_rank_fusion(dense_results, bm25_results, k: int = 60, top_k: int = 20):
        scores = {}

        for rank, result in enumerate(dense_results):
            doc_id = result.id
            if doc_id not in scores:
                scores[doc_id] = {"score": 0, "payload": result.payload}
            scores[doc_id]["score"] += 1 / (k + rank + 1)

        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            if doc_id not in scores:
                scores[doc_id] = {"score": 0, "payload": result["payload"]}
            scores[doc_id]["score"] += 1 / (k + rank + 1)

        sorted_results = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return sorted_results[:top_k]

    def rerank(self, query: str, fused_results, top_k: int = 8):
        pairs = [[query, result[1]["payload"]["text"]] for result in fused_results]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, fused_results), key=lambda x: x[0], reverse=True)
        return ranked[:top_k]

    def hybrid_search(self, query: str, top_k: int = 8):
        dense_results = self.dense_search(query, top_k=20)
        bm25_results = self.bm25_search(query, top_k=20)
        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)
        return self.rerank(query, fused, top_k=top_k)

    def image_search(self, query: str, top_k: int = 3):
        query_vector = self.embedding_model.encode(query).tolist()
        query_filter = None
        if self.image_whitelist is not None:
            query_filter = Filter(
                must=[FieldCondition(key="filename", match=MatchAny(any=list(self.image_whitelist)))]
            )

        return self.client.query_points(
            collection_name=self.image_collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points


def select_relevant_chunks(query: str, chunks: list[str], top_k: int = 10) -> list[str]:
    """Rank a small in-memory set of chunks (e.g. an uploaded document) by BM25 relevance to the
    query, best match first. Used instead of the full HybridRetriever pipeline since an uploaded
    document is a handful of chunks, not a Qdrant collection worth indexing."""
    if len(chunks) <= top_k:
        return chunks

    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[idx] for idx in top_indices]

