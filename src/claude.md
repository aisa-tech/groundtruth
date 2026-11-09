# Groundtruth

Retrieval evaluation harness for a legal research assistant. Portfolio project.
Corpus: 150 CUAD contracts (data/corpus.jsonl). Raw data: data/CUADv1.json (not in git).

## Rules
- Work through TODO.md in order. Do ONE unchecked task, then stop and summarize.
- After each task: run the check listed in the task, show me the output, tick the box.
- Tasks marked [HUMAN] are mine. Do not do them. Tell me what to do and stop.
- Never edit data/golden_v1.jsonl without asking me.
- Gold labels are character spans (doc_id, start, end), never chunk ids.
- Python 3.11, venv in .venv. Run commands from repo root.
- Explain every file you create in 2 to 3 sentences so I can defend it in an interview.