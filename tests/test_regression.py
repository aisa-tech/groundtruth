# tests/test_regression.py
import json
from src.run_eval import main

TOLERANCE = 0.01

def test_recall_did_not_regress():
    base = json.load(open("results/baseline_metrics.json"))
    main("configs/production.json")
    new = json.load(open("results/production.json"))
    drop = base["recall@10"] - new["recall@10"]
    assert drop <= TOLERANCE, (
        f"recall@10 dropped {drop:.3f} "
        f"({base['recall@10']:.3f} -> {new['recall@10']:.3f})")

def test_no_category_collapsed():
    base = json.load(open("results/baseline_metrics.json"))["by_category"]
    new = json.load(open("results/production.json"))["by_category"]
    for cat, val in base.items():
        assert new[cat] >= val - 0.05, f"{cat} recall fell from {val} to {new[cat]}"