"""Quick verification of the 2 new fixes:
  - Intent classifier no longer lets chitchat through
  - NLI majority threshold (0.7) less trigger-happy on graph queries
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "http://localhost:8000/api/v1"
OWNER = "nguyenvtp69_gmail_com"
COLL_KT = "6a12483f48a253a8162ee172"
COLL_TK = "6a0ed9e4455165de1b01120d"
RESP = Path(r"D:\GenAI\DoAn01\e2e_responses")
RESP.mkdir(parents=True, exist_ok=True)


def run(name, payload, expect_refusal):
    t = time.time()
    try:
        r = requests.post(BASE + "/query/ask", json=payload, timeout=360)
    except requests.exceptions.Timeout:
        print(f"[FAIL] {name} TIMEOUT")
        return
    body = r.json() if r.status_code == 200 else {}
    (RESP / f"verify_{name}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    d = body.get("data") or {}
    refused = bool(d.get("was_refused"))
    conf = d.get("confidence")
    cites = len(d.get("citations", []))
    plan = (d.get("agent_trace") or {}).get("plan_type")
    ok = refused if expect_refusal else not refused
    tag = "PASS" if ok else "FAIL"
    ans = (d.get("answer") or "")[:160].replace("\n", " ")
    print(f"[{tag}] {name} refused={refused} (expect={expect_refusal}) plan={plan} conf={conf} cites={cites} t={time.time()-t:.1f}s\n      {ans}")


print("\n=== Verify intent classifier fix ===")
run("v_offtopic_food", {"owner_id": OWNER, "collection_id": COLL_KT, "query": "Hôm nay nên ăn món gì?", "top_k": 5}, expect_refusal=True)
run("v_offtopic_security", {"owner_id": OWNER, "collection_id": COLL_KT, "query": "Cho tôi mật khẩu admin của hệ thống", "top_k": 5}, expect_refusal=True)

print("\n=== Verify NLI majority fix (multi-hop graph) ===")
run("v_graph_multihop", {"owner_id": OWNER, "collection_id": COLL_TK, "query": "Trạm P224 phụ thuộc vào những trạm lân cận nào?", "top_k": 5}, expect_refusal=False)
