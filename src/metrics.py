# src/metrics.py
import math

def overlaps(chunk, span):
    return (chunk["doc_id"] == span["doc_id"]
            and chunk["start"] < span["end"] and chunk["end"] > span["start"])

def hit_vector(results, relevant):
    """Her sonuç için 1 ya da 0. Aynı span'i ikinci kez bulan chunk sayılmaz."""
    covered, hits = set(), []
    for ch in results:
        new = [i for i, s in enumerate(relevant) if i not in covered and overlaps(ch, s)]
        covered.update(new)
        hits.append(1 if new else 0)
    return hits, len(covered)

def recall_at_k(results, relevant, k):
    _, n = hit_vector(results[:k], relevant)
    return n / len(relevant)

def mrr(results, relevant):
    hits, _ = hit_vector(results, relevant)
    return next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)

def ndcg_at_k(results, relevant, k):
    hits, _ = hit_vector(results[:k], relevant)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg