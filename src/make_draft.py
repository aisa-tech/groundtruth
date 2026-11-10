import json, random

random.seed(42)
PER_CAT = 20

rows = [json.loads(l) for l in open("data/candidates.jsonl")]
by_cat = {}
for r in rows:
    by_cat.setdefault(r["category"], []).append(r)

with open("data/golden_draft.jsonl", "w") as f:
    for cat, items in sorted(by_cat.items()):
        for i, r in enumerate(random.sample(items, PER_CAT), 1):
            f.write(json.dumps({
                "qid": f"{cat}-{i:03d}",
                "category": cat,
                "query": "",
                "answer_preview": r["answer"],
                "relevant": [{"doc_id": r["doc_id"], "start": r["start"], "end": r["end"]}],
                "reviewed": False}) + "\n")
print("written: data/golden_draft.jsonl")