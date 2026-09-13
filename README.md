# Multimodal-RAG-From-Scratch — MedLens

Bilingual (FR/EN) medical RAG chatbot. Live demo: https://multimodal-rag-from-scratch.vercel.app

## 📊 Evaluation Results (v1.0)

| Metric | Score | Target |
|---|---|---|
| Faithfulness | 0.70 | > 0.80 |
| Answer Relevancy | 0.61 | > 0.85 |
| Context Precision | 0.60 | > 0.70 |

*Evaluated on 19 bilingual medical questions (EN/FR)*
*Model: Llama 3.1 8B via Groq | Embeddings: all-MiniLM-L6-v2*
*v2.0 improvements planned: better chunking, larger dataset*

> ⚠️ **These numbers are from the original v1.0 pipeline and are now stale.** They were measured
> with Llama 3.1 8B and the full hybrid dense+BM25+cross-encoder-reranking pipeline. Since then:
> Llama 3.1 8B was retired from Groq (the app now uses `openai/gpt-oss-20b`), the cross-encoder
> reranker was removed, and the deployed backend runs dense-only retrieval (no BM25) to fit a
> free-tier memory budget. These scores do **not** reflect the currently deployed app — a
> re-evaluation on the current pipeline is planned before any v2.0 numbers are reported.
