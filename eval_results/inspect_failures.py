import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

lines = Path("eval_results/e2e_eval.jsonl").read_text(encoding="utf-8").strip().splitlines()
rows = {r["id"]: r for r in (json.loads(l) for l in lines if l.strip())}

targets = ["q011", "q012", "q013", "q020"]
for tid in targets:
    r = rows[tid]
    ans = (r.get("answer") or "")[:400]
    print(f"=== {tid} / {r['query_type']} ===")
    print("Q:", r["query"])
    print("A:", ans)
    print(f"refused={r.get('refused')} | relevance={r.get('answer_relevance')} | faithfulness={r.get('faithfulness')} | sem_faith={r.get('semantic_faithfulness')}")
    cits = r.get("citations") or []
    snippets = [c.get("snippet_original", "")[:80] for c in cits[:2]]
    print(f"citations={len(cits)}: {snippets}")
    print()
