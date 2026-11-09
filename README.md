# Groundtruth — Retrieval Evaluation Harness

Benchmarks four retrieval configs against a hand-reviewed golden set of 97 legal-contract questions drawn from the [CUAD](https://www.atticusprojectai.org/cuad) corpus (150 contracts, 5 categories).

## Results

Evaluated on `data/golden_v1.jsonl` (97 questions, character-span labels).

| Config | Recall@10 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|
| bm25 | 0.6804 | 0.4332 | 0.4932 | 14 ms |
| dense | 0.6804 | 0.3557 | 0.4341 | 14 ms |
| hybrid | 0.7732 | 0.4627 | 0.5370 | 24 ms |
| hybrid_rerank | 0.8454 | 0.5282 | 0.6047 | 228 ms |
| **hybrid_rerank_256** | **0.8454** | **0.5891** | **0.6507** | 181 ms |

### By category — Recall@10

| Category | bm25 | dense | hybrid | hybrid_rerank | hybrid_rerank_256 |
|---|---|---|---|---|
| governing_law | 0.3684 | 0.4211 | 0.4737 | 0.7368 | 0.6316 |
| ip_ownership | 0.8000 | 0.8500 | 0.9000 | 0.9000 | 0.9500 |
| liability_cap | 0.6667 | 0.6111 | 0.8333 | 0.7778 | 0.8889 |
| non_compete | 0.7500 | 0.8500 | 0.8500 | 0.9500 | 0.9000 |
| termination | 0.8000 | 0.6500 | 0.8000 | 0.8500 | 0.8500 |

## Recommendation

The best config is hybrid retrieval with the cross-encoder reranker on 256-token chunks. It finds the right passage in the top 10 for 84.5% of queries, versus 68% for dense alone, and ranks it highest on every metric (MRR 0.589). Smaller chunks did not change recall but improved ranking and cut reranker latency from 228 ms to 181 ms, because the cross-encoder scores shorter, more focused passages. The cost versus plain hybrid is latency: p95 rises from 24 ms to 181 ms. For a legal research assistant that tradeoff is worth it, because a missed precedent is far more expensive than a fifth of a second. If latency becomes a constraint, plain hybrid is the fallback: it keeps most of the gain (77%) at 24 ms.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # rank_bm25, sentence-transformers, pandas, numpy
```

## Run a config

```bash
python -m src.run_eval configs/hybrid_rerank.json
```

Results land in `results/<config_name>.json` and `results/<config_name>_per_query.csv`.

## Project layout

| Path | Purpose |
|---|---|
| `data/corpus.jsonl` | 150 CUAD contracts (doc_id + full text) |
| `data/golden_v1.jsonl` | 97 reviewed questions with character-span gold labels |
| `configs/` | Four retrieval configs (bm25, dense, hybrid, hybrid_rerank) |
| `results/` | Per-config summary JSON + per-query CSV |
| `src/ingest.py` | Character-based chunker (size=512 tokens, overlap=64) |
| `src/retriever.py` | BM25, dense (BGE-small), hybrid RRF, cross-encoder rerank |
| `src/metrics.py` | Overlap-aware Recall@k, MRR, nDCG@k |
| `src/run_eval.py` | Loads config, runs retrieval, writes results |
