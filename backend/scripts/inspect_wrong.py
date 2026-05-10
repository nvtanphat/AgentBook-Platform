import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

samples = [json.loads(l) for l in Path("eval_results/retrieval_eval_v2.jsonl").open(encoding="utf-8") if l.strip()]
wrong = [s for s in samples if s.get("retrieval_ok") == "wrong"]
for s in wrong:
    print(f'[{s["query_type"]}] {s["query"]}')
    for c in s["retrieved_chunks"][:3]:
        print(f'  {c["document_name"]}')
        print(f'  preview: {c["content_preview"][:180]}')
    print()
