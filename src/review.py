import json, os

DRAFT = "data/golden_draft.jsonl"
OUT = "data/golden_v1.jsonl"
LOG = "data/review_log.json"

rows = [json.loads(l) for l in open(DRAFT)]
log = json.load(open(LOG)) if os.path.exists(LOG) else {"accepted": 0, "edited": 0, "deleted": 0}

def save():
    with open(DRAFT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(log, open(LOG, "w"), indent=2)

todo = [r for r in rows if not r.get("reviewed") and not r.get("deleted")]
print(f"{len(todo)} questions left. y = accept, e = edit question, d = delete, q = quit and save\n")

for i, r in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {r['qid']}  ({r['category']})")
    print("Question :", r["query"])
    print("Answer:", " ".join(r["answer_preview"].split()))
    while True:
        c = input("> ").strip().lower()
        if c == "y":
            r["reviewed"] = True; log["accepted"] += 1; break
        if c == "e":
            r["query"] = input("New question: ").strip()
            r["reviewed"] = True; log["edited"] += 1; break
        if c == "d":
            r["deleted"] = True; log["deleted"] += 1; break
        if c == "q":
            save(); print("Saved. will resume from where you left off."); raise SystemExit
    save()
    print()

final = [r for r in rows if r.get("reviewed") and not r.get("deleted")]
with open(OUT, "w") as f:
    for r in final:
        f.write(json.dumps({k: r[k] for k in ("qid", "category", "query", "relevant", "reviewed")}) + "\n")
print(f"Done. {len(final)} questions written to {OUT}. Summary: {log}")