# src/run_eval.py
import json, sys, time
import numpy as np, pandas as pd
from src.ingest import chunk
from src.retriever import Retriever
from src.metrics import recall_at_k, mrr, ndcg_at_k

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def main(cfg_path, golden="data/golden_v1.jsonl"):
    cfg = json.load(open(cfg_path))
    chunks = [dict(c, doc_id=d["doc_id"])
              for d in load_jsonl("data/corpus.jsonl")
              for c in chunk(d["text"], size=cfg["chunk_size"])]
    r = Retriever(chunks, mode=cfg["mode"], rerank=cfg["rerank"])
    rows = []
    for q in load_jsonl(golden):
        t = time.perf_counter()
        res = r.search(q["query"], k=10)
        ms = (time.perf_counter() - t) * 1000
        rows.append({"qid": q["qid"], "category": q["category"], "latency_ms": ms,
                     "recall@10": recall_at_k(res, q["relevant"], 10),
                     "mrr": mrr(res, q["relevant"]),
                     "ndcg@10": ndcg_at_k(res, q["relevant"], 10)})
    df = pd.DataFrame(rows)
    summary = {"config": cfg["name"],
               **df[["recall@10", "mrr", "ndcg@10"]].mean().round(4).to_dict(),
               "p95_latency_ms": round(float(np.percentile(df.latency_ms, 95)), 1),
               "by_category": df.groupby("category")["recall@10"].mean().round(4).to_dict()}
    df.to_csv(f"results/{cfg['name']}_per_query.csv", index=False)
    json.dump(summary, open(f"results/{cfg['name']}.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main(sys.argv[1])