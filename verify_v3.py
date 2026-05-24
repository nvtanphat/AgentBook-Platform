"""Verify fixes after v3:
  3.4 cross-lingual EN→VN — claim_verifier skipped on language mismatch
  1.1 indexing reproducibility — fresh upload + index
"""
from __future__ import annotations
import json, time, requests, uuid
from pathlib import Path

BASE = "http://localhost:8000/api/v1"
OWNER = "nguyenvtp69_gmail_com"
COLL_TK = "6a0ed9e4455165de1b01120d"

# 3.4
print("\n=== 3.4 Cross-lingual EN→VN ===")
t = time.time()
p = {"owner_id": OWNER, "collection_id": COLL_TK,
     "query": "Why was WAPE chosen instead of MAPE?", "top_k": 5,
     "answer_language": "en"}
r = requests.post(BASE + "/query/ask", json=p, timeout=360)
if r.status_code == 200:
    d = r.json().get("data") or {}
    ans = d.get("answer", "")[:300].replace("\n", " ")
    ok = not d.get("was_refused") and len(d.get("citations", [])) > 0
    print(f"[{'PASS' if ok else 'FAIL'}] refused={d.get('was_refused')} cites={len(d.get('citations',[]))} lang={d.get('answer_language')} t={time.time()-t:.1f}s")
    print(f"  ANSWER: {ans}")
else:
    print(f"[FAIL] http={r.status_code} body={r.text[:200]}")

# 1.1 reproducibility — fresh upload + index
print("\n=== 1.1 Indexing flow (fresh upload) ===")
t = time.time()
USER = f"verify_{uuid.uuid4().hex[:6]}"
DOC = Path(r"D:\GenAI\DoAn01\data\test data\rag_mau_hoc_tap.pdf")
rc = requests.post(BASE + "/collections", json={"owner_id": USER, "name": "verify"})
if rc.status_code not in (200, 201):
    print(f"[FAIL] collection create http={rc.status_code}")
else:
    coll_id = rc.json()["data"]["collection_id"]
    with DOC.open("rb") as fh:
        meta = json.dumps({"owner_id": USER, "collection_id": coll_id, "language": "vi"})
        ru = requests.post(BASE + "/materials/upload", data={"metadata": meta},
                           files={"file": (DOC.name, fh, "application/pdf")}, timeout=60)
    if ru.status_code not in (200, 201):
        print(f"[FAIL] upload http={ru.status_code} body={ru.text[:300]}")
    else:
        mat_id = ru.json()["data"]["material_id"]
        print(f"  uploaded mat_id={mat_id}, polling...")
        indexed = False
        last_status = None
        for i in range(180):
            rs = requests.get(BASE + f"/materials/{mat_id}/status", params={"owner_id": USER})
            if rs.status_code == 200:
                last_status = rs.json().get("data", {}).get("status")
                if last_status == "indexed":
                    indexed = True
                    break
                if last_status == "failed":
                    break
            time.sleep(2)
        print(f"[{'PASS' if indexed else 'FAIL'}] final_status={last_status} elapsed={time.time()-t:.1f}s")
    # cleanup
    requests.delete(BASE + f"/collections/{coll_id}", params={"owner_id": USER})
