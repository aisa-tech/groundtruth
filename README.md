# Groundtruth — Retrieval Evaluation Harness

A retrieval evaluation harness for a legal research assistant. It benchmarks retrieval configs against a hand-reviewed golden set of 97 contract questions drawn from the [CUAD](https://www.atticusprojectai.org/cuad) corpus (150 contracts, 5 clause categories), and a CI gate fails any pull request that makes retrieval worse.

**Production config:** hybrid BM25 + dense retrieval, cross-encoder rerank, 256-token chunks prefixed with the document title.
**Recall@10 0.959 · MRR 0.823 · nDCG@10 0.857 · p95 169 ms**

## How it works

1. **Corpus.** 150 CUAD contracts in `data/corpus.jsonl`.
2. **Golden set.** 100 draft questions were written from CUAD answer spans, then reviewed by hand; 3 were rejected, leaving 97. Each gold label is a character span (`doc_id`, `start`, `end`), never a chunk id, so chunking can change without relabelling.
3. **Chunking.** Character windows of `chunk_size × 4` characters with 64-token overlap. Each chunk keeps its `start` and `end`.
4. **Retrieval.** BM25, dense (`BAAI/bge-small-en-v1.5`), hybrid via reciprocal rank fusion (k=60), and optional cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top 50 candidates.
5. **Metrics.** A retrieved chunk is a hit when it has the same `doc_id` as a gold span and the character ranges overlap. A span that is already covered is not counted again. From this: Recall@10, MRR and nDCG@10.

## Results

Evaluated on `data/golden_v1.jsonl` (97 questions).

| Config | Recall@10 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|
| bm25 | 0.6804 | 0.4332 | 0.4932 | 14 ms |
| dense | 0.6804 | 0.3557 | 0.4341 | 14 ms |
| hybrid | 0.7732 | 0.4627 | 0.5370 | 24 ms |
| hybrid_rerank | 0.8454 | 0.5282 | 0.6047 | 228 ms |
| hybrid_rerank_256 | 0.8454 | 0.5891 | 0.6507 | 181 ms |
| **production** (hybrid_rerank_256 + title prefix) | **0.9588** | **0.8226** | **0.8568** | 169 ms |

### By category — Recall@10

| Category | bm25 | dense | hybrid | hybrid_rerank | hybrid_rerank_256 | production |
|---|---|---|---|---|---|---|
| governing_law | 0.3684 | 0.4211 | 0.4737 | 0.7368 | 0.6316 | **0.8947** |
| ip_ownership | 0.8000 | 0.8500 | 0.9000 | 0.9000 | 0.9500 | **1.0000** |
| liability_cap | 0.6667 | 0.6111 | 0.8333 | 0.7778 | 0.8889 | **1.0000** |
| non_compete | 0.7500 | 0.8500 | 0.8500 | 0.9500 | 0.9000 | 0.9000 |
| termination | 0.8000 | 0.6500 | 0.8000 | 0.8500 | 0.8500 | **1.0000** |

## Recommendation

Written before the title-prefix experiment:

> The best config is hybrid retrieval with the cross-encoder reranker on 256-token chunks. It finds the right passage in the top 10 for 84.5% of queries, versus 68% for dense alone, and ranks it highest on every metric (MRR 0.589). Smaller chunks did not change recall but improved ranking and cut reranker latency from 228 ms to 181 ms, because the cross-encoder scores shorter, more focused passages. The cost versus plain hybrid is latency: p95 rises from 24 ms to 181 ms. For a legal research assistant that tradeoff is worth it, because a missed precedent is far more expensive than a fifth of a second. If latency becomes a constraint, plain hybrid is the fallback: it keeps most of the gain (77%) at 24 ms.

**Update.** Production now adds the document title to each chunk on top of that config. Recall@10 rises from 0.845 to 0.959 and MRR from 0.589 to 0.823, with no added latency.

## Improvement experiments

Starting from `hybrid_rerank_256`, each change was tested on its own:

| Change | Recall@10 | MRR | nDCG@10 | governing_law | p95 |
|---|---|---|---|---|---|
| none (hybrid_rerank_256) | 0.845 | 0.589 | 0.651 | 0.632 | 178 ms |
| bge-large embeddings | 0.835 | 0.591 | 0.650 | 0.632 | 172 ms |
| sentence-aware chunk boundaries | 0.794 | 0.586 | 0.636 | 0.421 | 170 ms |
| chunk size 128 | 0.742 | 0.587 | 0.625 | 0.368 | 174 ms |
| rerank pool 50 → 100 | 0.845 | 0.603 | 0.662 | 0.632 | 336 ms |
| MiniLM-L-12 reranker | 0.866 | 0.613 | 0.674 | 0.632 | 306 ms |
| **document title prefix** | **0.959** | **0.823** | **0.857** | **0.895** | 177 ms |

**Why the title prefix works.** A chunk from the middle of a contract rarely names the company or the contract type; only the first page does. Without the title, a governing-law clause from one sponsorship agreement looks the same as one from any other, so similar clauses from the wrong contracts filled the top 10. With the title in the chunk text, BM25, the embedder and the reranker can all match the company named in the question. Gold matching uses character offsets, so changing chunk text does not affect scoring.

**Caveat.** Every golden question names the company and contract type, because that is how the set was written. The gain applies to questions that identify the contract. Cross-contract questions ("which agreements have a non-compete?") would benefit much less, and the golden set does not measure them yet.

**What did not work.** Snapping chunk ends to sentence boundaries made chunk lengths uneven and pushed short governing-law clauses out of the windows that were retrieved. Chunks of 128 tokens had the same problem. bge-large was no better than bge-small on this corpus. A larger rerank pool or a deeper reranker gave small gains at almost twice the latency.

All settings are config keys (`title_prefix`, `sentence_aware`, `embed_model`, `rerank_model`, `rerank_pool`), so each row can be reproduced with `src/run_eval.py`.

## Error analysis

Before the title prefix there were 15 misses (recall 0 at k=10). Five representative cases:

| Query | Failure | Cause |
|---|---|---|
| governing_law-009 | Right contract, wrong chunk | The gold span ends 614 characters before the nearest retrieved chunk; a neighbouring chunk outranked the one that contains it. |
| non_compete-014 | Right contract, wrong chunk | The gold span is a 26-character list header ("The Distributor shall not:"). The chunks with the actual restrictions rank above it. |
| governing_law-011 | Wrong contract | The contract is titled "Photo Retouching Outsourcing Agreement"; other outsourcing agreements with clearer governing-law clauses ranked higher. |
| governing_law-017 | Right contract, wrong chunk | The clause sits at 93% of a 52K-character contract; the opening pages and the signature block ranked higher. |
| termination-016 | Right contract, wrong chunk | 141K-character contract; other termination clauses outranked the termination-for-convenience clause. |

The title prefix fixed 11 of the 15. Four remain: `governing_law-009`, `governing_law-015`, `non_compete-011` and `non_compete-014`. All four retrieve the right contract but not the chunk that holds a short or cross-referenced span. Clause-aware chunking on section numbers is the next thing to try.

## Answer quality (LLM judge)

`src/generate.py` answers 10 sampled questions from the top 5 chunks of `hybrid_rerank_256` and asks Claude to score each answer from 1 to 5 for faithfulness (every claim is supported by the passages) and relevance (the answer addresses the question). Results are in `results/generate_sample.json`: average faithfulness 3.5 and relevance 3.8. This run predates the title prefix. Ten questions are too few for firm conclusions, and the judge has not been checked against human ratings.

## CI regression gate

`results/baseline_metrics.json` holds the locked production metrics. On every pull request, `.github/workflows/eval.yml` runs `pytest tests/`, and `tests/test_regression.py` re-runs the production config and fails if:

- Recall@10 drops more than 1 point below the baseline, or
- any category's Recall@10 drops more than 5 points.

To check the gate, a demo PR turned the reranker off. CI failed as intended: recall@10 dropped from 0.845 to 0.711, and governing_law dropped from 0.632 to 0.421. The PR was closed without merging.

The HuggingFace model cache is keyed on `configs/production.json`, so a PR that changes that file downloads the models again and the check takes 20 to 40 minutes.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`src/generate.py` needs an Anthropic API key in a `.env` file at the repo root (already in `.gitignore`):

```
ANTHROPIC_API_KEY=...
```

## Usage

```bash
python -m src.run_eval configs/production.json   # writes results/production.json and _per_query.csv
python -m src.generate                           # LLM answers + judge on 10 sampled questions
pytest tests/ -v                                 # metric unit tests + regression gate
```

## Project layout

| Path | Purpose |
|---|---|
| `data/corpus.jsonl` | 150 CUAD contracts (doc_id + full text) |
| `data/golden_v1.jsonl` | 97 reviewed questions with character-span gold labels |
| `configs/` | Retrieval configs: bm25, dense, hybrid, hybrid_rerank, hybrid_rerank_256, production |
| `results/` | Per-config summary JSON, per-query CSV, locked `baseline_metrics.json`, LLM judge sample |
| `src/ingest.py` | Corpus builder and character-based chunker (optional sentence-aware boundaries) |
| `src/retriever.py` | BM25, dense (BGE-small), hybrid RRF, cross-encoder rerank |
| `src/metrics.py` | Overlap-aware Recall@k, MRR, nDCG@k |
| `src/run_eval.py` | Loads a config, builds chunks (optional title prefix), runs retrieval, writes results |
| `src/generate.py` | Answer generation and LLM-as-judge scoring |
| `tests/test_metrics.py` | Unit tests for the metrics |
| `tests/test_regression.py` | CI regression gate against the locked baseline |
| `.github/workflows/eval.yml` | Runs the tests on every pull request |
