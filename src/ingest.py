# src/ingest.py
import json

def build_corpus(n_docs=150, src="data/CUADv1.json", out="data/corpus.jsonl"):
    data = json.load(open(src))["data"]
    with open(out, "w") as f:
        for doc in data[:n_docs]:
            f.write(json.dumps({"doc_id": doc["title"],
                                "text": doc["paragraphs"][0]["context"]}) + "\n")

def chunk(text, size=512, overlap=64, sentence_aware=False):
    """Character-based chunker. Each chunk carries its start and end position.

    When sentence_aware=True the window boundary snaps to the nearest sentence
    end (. ! ? followed by whitespace, or blank line) within ±200 chars so
    chunks do not cut mid-sentence.
    """
    win  = size * 4
    step = (size - overlap) * 4
    chunks, start = [], 0
    while start < len(text):
        end = min(start + win, len(text))
        if sentence_aware and end < len(text):
            end = _snap_sentence(text, end, slack=200)
        chunks.append({"start": start, "end": end, "text": text[start:end]})
        if end == len(text):
            break
        next_start = end - overlap * 4
        start = next_start if next_start > start else start + step
    return chunks


def _snap_sentence(text, pos, slack=200):
    """Return the position of the nearest sentence boundary around pos."""
    lo = max(0, pos - slack)
    hi = min(len(text), pos + slack)
    segment = text[lo:hi]
    best, best_dist = pos, slack + 1
    for i, ch in enumerate(segment):
        abs_i = lo + i
        if ch in '.!?' and abs_i + 1 < len(text) and text[abs_i + 1] in ' \n\t':
            d = abs(abs_i + 1 - pos)
            if d < best_dist:
                best, best_dist = abs_i + 1, d
        elif ch == '\n' and abs_i + 1 < len(text) and text[abs_i + 1] == '\n':
            d = abs(abs_i + 1 - pos)
            if d < best_dist:
                best, best_dist = abs_i + 2, d
    return best