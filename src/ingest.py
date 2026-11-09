# src/ingest.py
import json

def build_corpus(n_docs=150, src="data/CUADv1.json", out="data/corpus.jsonl"):
    data = json.load(open(src))["data"]
    with open(out, "w") as f:
        for doc in data[:n_docs]:
            f.write(json.dumps({"doc_id": doc["title"],
                                "text": doc["paragraphs"][0]["context"]}) + "\n")

def chunk(text, size=512, overlap=64):
    """Karakter bazlı chunk. Her chunk başlangıç ve bitiş konumunu taşır."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size * 4, len(text))  # ~4 karakter = 1 token
        chunks.append({"start": start, "end": end, "text": text[start:end]})
        if end == len(text):
            break
        start = end - overlap * 4
    return chunks