"""Verify the 3 quality fixes: SLEC stricter, prompt verbatim, language drift strip."""
import json, time, requests

BASE = "http://localhost:8000/api/v1"
OWNER = "nguyenvtp69_gmail_com"
COLL_TK = "6a0ed9e4455165de1b01120d"
COLL_KT = "6a12483f48a253a8162ee172"

cases = [
    ("kan_gru_vi", {"owner_id": OWNER, "collection_id": COLL_TK, "query": "KAN liên quan đến GRU như thế nào?", "top_k": 5}),
    ("p224_multihop", {"owner_id": OWNER, "collection_id": COLL_TK, "query": "Trạm P224 phụ thuộc vào những trạm lân cận nào?", "top_k": 5}),
    ("wape_en", {"owner_id": OWNER, "collection_id": COLL_TK, "query": "Why was WAPE chosen instead of MAPE?", "top_k": 5, "answer_language": "en"}),
    ("rag_kt_definition", {"owner_id": OWNER, "collection_id": COLL_KT, "query": "RAG là gì và dùng để làm gì?", "top_k": 5}),
]

for name, payload in cases:
    t = time.time()
    print(f"\n=== {name} ===")
    try:
        r = requests.post(BASE + "/query/ask", json=payload, timeout=400)
    except Exception as e:
        print(f"  ERR: {e}")
        continue
    if r.status_code != 200:
        print(f"  http={r.status_code}")
        continue
    d = r.json().get("data") or {}
    refused = bool(d.get("was_refused"))
    cites = len(d.get("citations", []))
    conf = d.get("confidence")
    ans = (d.get("answer") or "").replace("\n", " ")
    print(f"  refused={refused} conf={conf} cites={cites} t={time.time()-t:.1f}s")
    print(f"  ANSWER: {ans[:400]}")
    sc = d.get("sentence_coverage") or {}
    if sc:
        sup = sum(1 for s in (sc.get("sentences") or []) if s.get("status") == "supported")
        par = sum(1 for s in (sc.get("sentences") or []) if s.get("status") == "partial")
        uns = sum(1 for s in (sc.get("sentences") or []) if s.get("status") == "unsupported")
        print(f"  SLEC: supported={sup} partial={par} unsupported={uns} dropped={sc.get('dropped_count')}")
