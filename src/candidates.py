# src/candidates.py
import json, re

CATS = {"governing law": "governing_law",
        "termination for convenience": "termination",
        "cap on liability": "liability_cap",
        "non-compete": "non_compete",
        "ip ownership assignment": "ip_ownership"}

docs = {json.loads(l)["doc_id"] for l in open("data/corpus.jsonl")}
data = json.load(open("data/CUADv1.json"))["data"]

with open("data/candidates.jsonl", "w") as f:
    for doc in data:
        if doc["title"] not in docs:
            continue
        for qa in doc["paragraphs"][0]["qas"]:
            m = re.search(r'related to "([^"]+)"', qa["question"])
            cat = CATS.get(m.group(1).lower()) if m else None
            if not cat or not qa["answers"]:
                continue
            a = qa["answers"][0]
            f.write(json.dumps({"doc_id": doc["title"], "category": cat,
                                "answer": a["text"][:300], "start": a["answer_start"],
                                "end": a["answer_start"] + len(a["text"])}) + "\n")