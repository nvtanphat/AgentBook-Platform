import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

samples = [json.loads(l) for l in Path("eval_results/retrieval_eval.jsonl").open(encoding="utf-8") if l.strip()]
for s in samples:
    print(f'[{s["query_type"]}] {s["query"][:60]}  -> verdict={s["retrieval_ok"]}')
    for c in s["retrieved_chunks"][:2]:
        print(f'  doc={c["document_name"]}  score={c["score"]}')
        print(f'  preview: {c["content_preview"][:120]}')
    print()
