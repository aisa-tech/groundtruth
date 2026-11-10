# src/generate.py
import json, random, sys, anthropic
from dotenv import load_dotenv
from src.ingest import chunk
from src.retriever import Retriever

load_dotenv()  # loads .env from the repo root
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
MODEL = "claude-sonnet-5-5"

def _text(msg):
    return next(b.text for b in msg.content if hasattr(b, "text"))

def answer(query, chunks):
    ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks[:5]))
    msg = client.messages.create(model=MODEL, max_tokens=400, messages=[{
        "role": "user",
        "content": f"Answer only from the passages. Cite like [1]. "
                   f"If they do not contain the answer, say so.\n\n{ctx}\n\nQuestion: {query}"}])
    return _text(msg)

def judge(query, chunks, ans):
    ctx = "\n\n".join(c["text"] for c in chunks[:5])
    prompt = ("Score two things from 1 to 5 and reply as JSON "
              '{"faithfulness": n, "relevance": n, "reason": "..."}.\n'
              "faithfulness: is every claim in the answer supported by the passages?\n"
              "relevance: does the answer address the question?\n\n"
              f"Passages:\n{ctx}\n\nQuestion: {query}\n\nAnswer: {ans}")
    msg = client.messages.create(model=MODEL, max_tokens=300,
                                 messages=[{"role": "user", "content": prompt}])
    raw = _text(msg).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": 0, "relevance": 0, "reason": f"parse error: {raw[:80]}"}


def main(cfg_path="configs/hybrid_rerank_256.json",
         golden="data/golden_v1.jsonl",
         out="results/generate_sample.json",
         n=10, seed=42):
    cfg = json.load(open(cfg_path))
    corpus = [json.loads(l) for l in open("data/corpus.jsonl")]
    chunks = [dict(c, doc_id=d["doc_id"])
              for d in corpus
              for c in chunk(d["text"], size=cfg["chunk_size"])]
    retriever = Retriever(chunks, mode=cfg["mode"], rerank=cfg["rerank"])

    questions = [json.loads(l) for l in open(golden)]
    random.seed(seed)
    sample = random.sample(questions, min(n, len(questions)))

    rows = []
    for i, q in enumerate(sample, 1):
        print(f"[{i}/{n}] {q['qid']}", flush=True)
        results = retriever.search(q["query"], k=10)
        ans = answer(q["query"], results)
        scores = judge(q["query"], results, ans)
        rows.append({"qid": q["qid"], "category": q["category"],
                     "query": q["query"], "answer": ans, **scores})

    summary = {"n": len(rows),
               "avg_faithfulness": round(sum(r["faithfulness"] for r in rows) / len(rows), 2),
               "avg_relevance": round(sum(r["relevance"] for r in rows) / len(rows), 2),
               "rows": rows}
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nwritten: {out}")
    print(f"avg faithfulness: {summary['avg_faithfulness']}  "
          f"avg relevance: {summary['avg_relevance']}")


if __name__ == "__main__":
    main(*sys.argv[1:])