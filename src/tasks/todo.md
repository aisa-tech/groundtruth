# TODO

- [x] Setup, venv, folders
- [x] Corpus: data/corpus.jsonl (150 docs)
- [x] Candidates: data/candidates.jsonl (326 rows)
- [x] Draft golden set: src/make_draft.py -> data/golden_draft.jsonl (check: 100 lines)
- [x] Fill "query" fields in golden_draft.jsonl (natural lawyer questions, no copied phrases)
- [x] [HUMAN] Review all 100 queries, set reviewed=true, count rejected drafts, save as data/golden_v1.jsonl
- [x] src/metrics.py + tests/test_metrics.py (check: pytest tests/test_metrics.py passes)
- [x] src/run_eval.py + 4 configs in configs/ (check: baseline runs, prints recall@10)
- [x] Run all 4 configs, build results table in README
- [x] [HUMAN] Write the recommendation paragraph with the tradeoff
- [x] src/generate.py: answer + LLM judge on 10 queries (needs ANTHROPIC_API_KEY)
- [x] CI: production config, results/baseline_metrics.json, tests/test_regression.py, .github/workflows/eval.yml
- [ ] [HUMAN] Open a bad-change PR and screenshot the red check
- [ ] [HUMAN] Error analysis: 5 failing queries and why
- [ ] README final pass