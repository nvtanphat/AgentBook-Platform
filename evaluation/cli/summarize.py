import json
from pathlib import Path

lines = Path("evaluation/results/e2e_eval.jsonl").read_text(encoding="utf-8").strip().splitlines()
rows = [json.loads(l) for l in lines if l.strip()]

def asc(k):
    return key_val(k)

def key_val(k):
    return lambda r: r.get(k) or 0.0

print("=== REFUSED ===")
for r in rows:
    if r.get("refused"):
        q = r["query"].encode("ascii", "replace").decode()[:60]
        print(f"  {r['id']}  {r['query_type']}  {q}")

print("\n=== WORST answer_relevance ===")
for r in sorted(rows, key=key_val("answer_relevance"))[:5]:
    q = r["query"].encode("ascii", "replace").decode()[:50]
    print(f"  {r.get('answer_relevance',0):.3f}  {r['id']}  {r['query_type']}  {q}")

print("\n=== WORST faithfulness ===")
for r in sorted(rows, key=key_val("faithfulness"))[:5]:
    q = r["query"].encode("ascii", "replace").decode()[:50]
    print(f"  {r.get('faithfulness',0):.3f}  {r['id']}  {r['query_type']}  {q}")

print("\n=== BEST answer_relevance ===")
for r in sorted(rows, key=key_val("answer_relevance"), reverse=True)[:3]:
    q = r["query"].encode("ascii", "replace").decode()[:50]
    print(f"  relevance={r.get('answer_relevance',0):.3f}  faith={r.get('faithfulness',0):.3f}  {q}")
