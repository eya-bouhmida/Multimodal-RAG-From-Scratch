from src.retriever import HybridRetriever, select_relevant_chunks


class FakePoint:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


def test_reciprocal_rank_fusion_combines_dense_and_bm25():
    dense = [FakePoint(1, {"text": "a"}), FakePoint(2, {"text": "b"})]
    bm25 = [
        {"id": 2, "score": 5.0, "payload": {"text": "b"}},
        {"id": 3, "score": 4.0, "payload": {"text": "c"}},
    ]

    fused = HybridRetriever.reciprocal_rank_fusion(dense, bm25, top_k=10)
    ids = [doc_id for doc_id, _ in fused]

    assert set(ids) == {1, 2, 3}
    # doc 2 appears in both lists (rank 1 in each), so it should come out on top
    assert ids[0] == 2


def test_select_relevant_chunks_returns_all_when_under_top_k():
    chunks = ["chunk one", "chunk two"]
    assert select_relevant_chunks("anything", chunks, top_k=5) == chunks


def test_select_relevant_chunks_ranks_by_bm25_relevance():
    chunks = [
        "The weather today is sunny and warm.",
        "Diabetes symptoms include thirst and fatigue.",
        "Cats are common household pets.",
    ]
    result = select_relevant_chunks("what are diabetes symptoms", chunks, top_k=1)
    assert result == ["Diabetes symptoms include thirst and fatigue."]
