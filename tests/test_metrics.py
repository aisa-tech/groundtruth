# tests/test_metrics.py
from src.metrics import recall_at_k, mrr, ndcg_at_k

SPAN = [{"doc_id": "d1", "start": 100, "end": 200}]
HIT = {"doc_id": "d1", "start": 150, "end": 400}
MISS = {"doc_id": "d2", "start": 0, "end": 300}

def test_hit_at_rank_2():
    res = [MISS, HIT]
    assert recall_at_k(res, SPAN, 10) == 1.0
    assert mrr(res, SPAN) == 0.5
    assert 0 < ndcg_at_k(res, SPAN, 10) < 1

def test_no_hit():
    assert recall_at_k([MISS], SPAN, 10) == 0.0
    assert mrr([MISS], SPAN) == 0.0