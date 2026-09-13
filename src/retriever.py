import json
from array import array
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from rank_bm25 import BM25Okapi
from scipy import sparse

# Curated list of image filenames worth showing (charts/graphs/medically meaningful
# illustrations) — most extracted images are logos, covers or icons and are excluded.
# See data/processed/image_whitelist.json.
IMAGE_WHITELIST_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "image_whitelist.json"


def _load_image_whitelist() -> set[str] | None:
    if not IMAGE_WHITELIST_PATH.exists():
        return None
    with open(IMAGE_WHITELIST_PATH, encoding="utf-8") as f:
        return set(json.load(f))


class _ChunkRecord:
    """Lightweight stand-in for a Qdrant point: only the fields build_prompt() actually reads
    (text, filename, page_num), instead of the full point object + payload for all 38k chunks."""

    __slots__ = ("id", "payload")

    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class CompactBM25:
    """Memory-efficient BM25Okapi (same formula/defaults as rank_bm25.BM25Okapi, verified to
    produce identical rankings on the real corpus) using a sparse term-document matrix instead
    of a fully materialized list of tokenized documents. For ~38k documents, naively building
    `[text.split() for text in texts]` before indexing costs several hundred MB in Python string/
    list overhead alone — this avoids ever holding that full structure in memory at once."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b

        vocab: dict[str, int] = {}
        rows = array("i")
        cols = array("i")
        data = array("f")
        doc_len = array("f")

        for doc_idx, text in enumerate(texts):
            counts: dict[int, int] = {}
            n_tokens = 0
            for word in text.lower().split():
                word_id = vocab.setdefault(word, len(vocab))
                counts[word_id] = counts.get(word_id, 0) + 1
                n_tokens += 1
            doc_len.append(n_tokens)
            for word_id, count in counts.items():
                rows.append(doc_idx)
                cols.append(word_id)
                data.append(count)

        n_docs = len(texts)
        n_vocab = len(vocab)
        tf = sparse.csr_matrix(
            (np.frombuffer(data, dtype=np.float32),
             (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
            shape=(n_docs, n_vocab),
            dtype=np.float32,
        )
        self.tf_csc = tf.tocsc()
        self.doc_len = np.frombuffer(doc_len, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if n_docs else 0.0
        self.corpus_size = n_docs
        self.vocab = vocab

        df = self.tf_csc.getnnz(axis=0).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        average_idf = idf.mean() if len(idf) else 0.0
        eps = epsilon * average_idf
        idf[idf < 0] = eps
        self.idf = idf

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        denom_base = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

        for word in tokens:
            word_id = self.vocab.get(word)
            if word_id is None:
                continue
            tf_col = self.tf_csc[:, word_id].toarray().ravel().astype(np.float64)
            idf = self.idf[word_id]
            scores += idf * (tf_col * (self.k1 + 1)) / (tf_col + denom_base)

        return scores


class HybridRetriever:
    """Dense (Qdrant) + BM25 hybrid search with RRF fusion."""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "medlens",
        image_collection_name: str = "medlens_images",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name
        self.image_collection_name = image_collection_name

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedding_model = TextEmbedding(model_name=embedding_model_name)
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
            for r in results:
                p = r.payload
                self._chunks.append(
                    _ChunkRecord(r.id, {"text": p["text"], "filename": p["filename"], "page_num": p["page_num"]})
                )
            if offset is None:
                break

        texts = [chunk.payload["text"] for chunk in self._chunks]
        self._bm25 = CompactBM25(texts)

    @property
    def text_chunks_count(self) -> int:
        return len(self._chunks)

    @property
    def image_chunks_count(self) -> int:
        return self.client.count(self.image_collection_name).count

    def _embed_query(self, query: str) -> list[float]:
        return list(self.embedding_model.embed([query]))[0].tolist()

    def dense_search(self, query: str, top_k: int = 20):
        query_vector = self._embed_query(query)
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

    def hybrid_search(self, query: str, top_k: int = 8):
        dense_results = self.dense_search(query, top_k=20)
        bm25_results = self.bm25_search(query, top_k=20)
        fused = self.reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)
        return [(item[1]["score"], item) for item in fused]

    def image_search(self, query: str, top_k: int = 3):
        query_vector = self._embed_query(query)
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
    query, best match first. Uses plain rank_bm25 (not CompactBM25) since an uploaded document is
    a handful of chunks, not large enough for memory efficiency to matter."""
    if len(chunks) <= top_k:
        return chunks

    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[idx] for idx in top_indices]
