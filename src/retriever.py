# src/retriever.py
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever:
    def __init__(self, chunks, mode="dense", rerank=False,
                 embed_model="BAAI/bge-small-en-v1.5",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.chunks, self.mode, self.rerank = chunks, mode, rerank
        texts = [c["text"] for c in chunks]
        self.embedder = SentenceTransformer(embed_model)
        self.emb = self.embedder.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=True)
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.reranker = CrossEncoder(rerank_model) if rerank else None

    def _dense(self, q, n):
        qv = self.embedder.encode([q], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.emb @ qv))[:n])

    def _bm25(self, q, n):
        return list(np.argsort(-self.bm25.get_scores(q.lower().split()))[:n])

    def _rrf(self, lists, k=60):
        """Reciprocal rank fusion: iki sıralamayı skor ölçeğine bakmadan birleştirir."""
        score = {}
        for lst in lists:
            for rank, idx in enumerate(lst):
                score[idx] = score.get(idx, 0) + 1 / (k + rank + 1)
        return sorted(score, key=score.get, reverse=True)

    def search(self, q, k=10):
        pool = 50 if self.rerank else k
        if self.mode == "dense":
            ids = self._dense(q, pool)
        elif self.mode == "bm25":
            ids = self._bm25(q, pool)
        else:  # hybrid
            ids = self._rrf([self._dense(q, 50), self._bm25(q, 50)])[:pool]
        if self.rerank:
            scores = self.reranker.predict([(q, self.chunks[i]["text"]) for i in ids])
            ids = [ids[i] for i in np.argsort(-scores)]
        return [self.chunks[i] for i in ids[:k]]