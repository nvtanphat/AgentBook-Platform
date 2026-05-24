"""E2E v3 — broader coverage across 4 phases (~85 min).

Phase 1 Critical:  upload+index, streaming, auth, conversation memory, ask-graph
Phase 2 Important: image-query, rate limit, material lifecycle, subgraph, evidence drilldown, collection CRUD
Phase 3 Edge:      concurrent queries, long conversation, empty collection, cross-lingual, reranker ablation
Phase 4 Defensive: path traversal, MIME mismatch, oversize, SQL inj, oversized query
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:8000/api/v1"
OWNER = "nguyenvtp69_gmail_com"
COLL_KT = "6a12483f48a253a8162ee172"   # Kiểm Thử (pre-indexed)
COLL_TK = "6a0ed9e4455165de1b01120d"   # ST-TopoKAN
MAT_LECTURE = "6a1248b548a253a8162ee173"

E2E_OWNER = "e2e_v3_user"  # isolated for create/delete tests
DOC_PATH = Path(r"D:\GenAI\DoAn01\data\test data\rag_mau_hoc_tap.pdf")
IMG_PATH = Path(r"D:\GenAI\DoAn01\data\test data\ML_Metrics_CheatSheet.png")

RESP = Path(r"D:\GenAI\DoAn01\e2e_responses\v3")
RESP.mkdir(parents=True, exist_ok=True)

TIMEOUT_LONG = 360
TIMEOUT_SHORT = 30

results: dict[str, tuple[bool, str]] = {}


def step(name, ok, detail=""):
    results[name] = (ok, detail)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name} — {detail}")
    sys.stdout.flush()


def trunc(s, n=150):
    if s is None:
        return ""
    return str(s).replace("\n", " ")[:n]


def save(name, data):
    try:
        (RESP / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def post(path, payload=None, timeout=TIMEOUT_SHORT):
    return requests.post(BASE + path, json=payload, timeout=timeout)


def get(path, params=None, timeout=TIMEOUT_SHORT):
    return requests.get(BASE + path, params=params, timeout=timeout)


# ────────────────────────────────────────────────────────────────────────
# Phase 1 — Critical (5 tests)
# ────────────────────────────────────────────────────────────────────────
print("\n=== Phase 1: Critical ===")

# 1.1 Upload + index flow
t = time.time()
coll_id = None
try:
    r = post("/collections", {"owner_id": E2E_OWNER, "name": "E2E v3"})
    coll_id = r.json()["data"]["collection_id"] if r.status_code in (200, 201) else None
    if not coll_id:
        step("1.1_upload_index", False, f"collection create failed http={r.status_code}")
    else:
        with DOC_PATH.open("rb") as fh:
            metadata = json.dumps({"owner_id": E2E_OWNER, "collection_id": coll_id, "language": "vi"})
            r = requests.post(BASE + "/materials/upload", data={"metadata": metadata},
                              files={"file": (DOC_PATH.name, fh, "application/pdf")}, timeout=60)
        if r.status_code not in (200, 201):
            step("1.1_upload_index", False, f"upload http={r.status_code} body={trunc(r.text)}")
        else:
            mat_id = r.json()["data"]["material_id"]
            indexed = False
            for _ in range(180):
                rs = get(f"/materials/{mat_id}/status")
                if rs.status_code == 200 and rs.json().get("data", {}).get("status") == "indexed":
                    indexed = True
                    break
                if rs.status_code == 200 and rs.json().get("data", {}).get("status") == "failed":
                    break
                time.sleep(2)
            step("1.1_upload_index", indexed, f"mat_id={mat_id} indexed={indexed} t={time.time()-t:.1f}s")
except Exception as e:
    step("1.1_upload_index", False, f"exc={e}")

# 1.2 Streaming SSE
t = time.time()
try:
    payload = {"owner_id": OWNER, "collection_id": COLL_KT, "query": "Module Docling parser xử lý loại file nào?", "top_k": 3}
    with requests.post(BASE + "/query/ask-stream", json=payload, timeout=TIMEOUT_LONG, stream=True) as r:
        chunks = []
        events = []
        for line in r.iter_lines():
            if line:
                ln = line.decode("utf-8", errors="ignore")
                if ln.startswith("event:"):
                    events.append(ln.split(":", 1)[1].strip())
                if ln.startswith("data:") and chunks is not None:
                    chunks.append(ln[5:].strip())
        ok = r.status_code == 200 and len(chunks) > 1
        step("1.2_stream", ok, f"http={r.status_code} events={events[:5]} chunk_count={len(chunks)} t={time.time()-t:.1f}s")
except Exception as e:
    step("1.2_stream", False, f"exc={e}")

# 1.3 Auth flow (register/login/me)
t = time.time()
import uuid
test_user_email = f"e2e_{uuid.uuid4().hex[:8]}@test.local"
try:
    r = post("/auth/register", {"email": test_user_email, "password": "TestPass123!"})
    if r.status_code not in (200, 201):
        step("1.3_auth", False, f"register http={r.status_code} body={trunc(r.text)}")
    else:
        token = r.json().get("data", {}).get("access_token")
        if not token:
            step("1.3_auth", False, "no access_token in register response")
        else:
            r2 = requests.get(BASE + "/auth/me", headers={"Authorization": f"Bearer {token}"}, timeout=10)
            ok = r2.status_code == 200
            step("1.3_auth", ok, f"register OK token=...{token[-10:]} me={r2.status_code} t={time.time()-t:.1f}s")
except Exception as e:
    step("1.3_auth", False, f"exc={e}")

# 1.4 Conversation memory (turn 1 answers; turn 2 uses "nó" anaphora)
t = time.time()
try:
    conv_id = f"e2e_conv_{uuid.uuid4().hex[:6]}"
    p1 = {"owner_id": OWNER, "collection_id": COLL_KT, "query": "Docling parser là gì?", "conversation_id": conv_id, "top_k": 3}
    r1 = post("/query/ask", p1, timeout=TIMEOUT_LONG)
    p2 = {"owner_id": OWNER, "collection_id": COLL_KT, "query": "Nó xử lý loại file nào?", "conversation_id": conv_id, "top_k": 3}
    r2 = post("/query/ask", p2, timeout=TIMEOUT_LONG)
    if r2.status_code == 200:
        d = r2.json().get("data") or {}
        ans = d.get("answer", "") or ""
        # If anaphora was resolved, answer should reference Docling/parser/PDF/DOCX
        has_docling = any(kw in ans.lower() for kw in ["docling", "pdf", "docx", "pptx"])
        save("1.4_conversation", {"turn1": r1.json(), "turn2": r2.json()})
        step("1.4_conversation", has_docling, f"anaphora resolved={has_docling} t={time.time()-t:.1f}s | {trunc(ans, 150)}")
    else:
        step("1.4_conversation", False, f"turn2 http={r2.status_code}")
except Exception as e:
    step("1.4_conversation", False, f"exc={e}")

# 1.5 ask-graph (graph-anchored query) — pick an entity from ST-TopoKAN graph
t = time.time()
try:
    rg = post("/graph", {"owner_id": OWNER, "collection_id": COLL_TK})
    nodes = (rg.json().get("data") or {}).get("nodes", [])
    kan_node = next((n for n in nodes if "kan" in (n.get("label") or "").lower()), None)
    if not kan_node:
        step("1.5_ask_graph", False, "no KAN-related node in graph response")
    else:
        payload = {"owner_id": OWNER, "collection_id": COLL_TK, "query": "Mô tả vai trò của entity này",
                   "entity_ids": [kan_node["id"]], "top_k": 3}
        r = post("/query/ask-graph", payload, timeout=TIMEOUT_LONG)
        ok = r.status_code == 200 and (r.json().get("data") or {}).get("answer", "").strip()
        save("1.5_ask_graph", r.json() if r.status_code == 200 else {"status": r.status_code, "body": r.text[:500]})
        step("1.5_ask_graph", ok, f"http={r.status_code} anchor={kan_node.get('label')} t={time.time()-t:.1f}s")
except Exception as e:
    step("1.5_ask_graph", False, f"exc={e}")


# ────────────────────────────────────────────────────────────────────────
# Phase 2 — Important (6 tests)
# ────────────────────────────────────────────────────────────────────────
print("\n=== Phase 2: Important ===")

# 2.1 image-as-query (multipart)
t = time.time()
try:
    if not IMG_PATH.exists():
        step("2.1_image_query", False, f"image not found: {IMG_PATH}")
    else:
        with IMG_PATH.open("rb") as fh:
            data = {"owner_id": OWNER, "collection_id": COLL_KT, "query_text": "Hình này nói về gì?"}
            r = requests.post(BASE + "/query/ask-image", data=data, files={"image": (IMG_PATH.name, fh, "image/png")}, timeout=TIMEOUT_LONG)
        ok = r.status_code == 200 and (r.json().get("data") or {}).get("answer", "").strip()
        save("2.1_image_query", r.json() if r.status_code == 200 else {"status": r.status_code, "body": r.text[:500]})
        step("2.1_image_query", ok, f"http={r.status_code} t={time.time()-t:.1f}s")
except Exception as e:
    step("2.1_image_query", False, f"exc={e}")

# 2.2 Rate limit (16 sequential within 60s)
t = time.time()
try:
    statuses = []
    for i in range(17):
        r = post("/query/ask", {"owner_id": OWNER, "collection_id": COLL_KT, "query": f"test {i}", "top_k": 1}, timeout=15)
        statuses.append(r.status_code)
        if r.status_code == 429:
            break
    hit_429 = 429 in statuses
    step("2.2_rate_limit", hit_429, f"got_429={hit_429} statuses={statuses[:18]} t={time.time()-t:.1f}s")
except Exception as e:
    step("2.2_rate_limit", False, f"exc={e}")

# Wait 60s for rate limit window to reset before continuing other tests
print("  ...waiting 60s for rate limit reset...")
time.sleep(60)

# 2.3 Material lifecycle (debug + raw + retry)
t = time.time()
try:
    rd = get(f"/materials/{MAT_LECTURE}/debug")
    debug_ok = rd.status_code == 200 and rd.json().get("success")
    rr = get(f"/materials/{MAT_LECTURE}/raw")
    raw_ok = rr.status_code == 200 and len(rr.content) > 100
    step("2.3_material_lifecycle", debug_ok and raw_ok, f"debug={rd.status_code} raw_bytes={len(rr.content) if rr.status_code==200 else 0}")
except Exception as e:
    step("2.3_material_lifecycle", False, f"exc={e}")

# 2.4 Subgraph endpoint
t = time.time()
try:
    rg = post("/graph", {"owner_id": OWNER, "collection_id": COLL_TK})
    nodes = (rg.json().get("data") or {}).get("nodes", [])
    if not nodes:
        step("2.4_subgraph", False, "no nodes in main graph")
    else:
        ent_id = nodes[0]["id"]
        rs = get(f"/graph/entity/{ent_id}/subgraph", params={"owner_id": OWNER, "collection_id": COLL_TK, "hops": 1})
        ok = rs.status_code == 200 and (rs.json().get("data") or {}).get("nodes")
        step("2.4_subgraph", ok, f"http={rs.status_code} anchor={ent_id} t={time.time()-t:.1f}s")
except Exception as e:
    step("2.4_subgraph", False, f"exc={e}")

# 2.5 Evidence drilldown
t = time.time()
try:
    re_ = get(f"/evidence/{MAT_LECTURE}/1")
    ok = re_.status_code == 200 and (re_.json().get("data") is not None)
    step("2.5_evidence", ok, f"http={re_.status_code} t={time.time()-t:.1f}s")
except Exception as e:
    step("2.5_evidence", False, f"exc={e}")

# 2.6 Collection CRUD (create/list/dashboard already covered; test dashboard)
t = time.time()
try:
    rd = get(f"/collections/{COLL_KT}/dashboard", params={"owner_id": OWNER})
    ok = rd.status_code == 200 and rd.json().get("success")
    step("2.6_collection_dashboard", ok, f"http={rd.status_code} t={time.time()-t:.1f}s")
except Exception as e:
    step("2.6_collection_dashboard", False, f"exc={e}")


# ────────────────────────────────────────────────────────────────────────
# Phase 3 — Edge (5 tests)
# ────────────────────────────────────────────────────────────────────────
print("\n=== Phase 3: Edge cases ===")

# 3.1 Concurrent queries (3 parallel)
t = time.time()
async def concurrent_calls():
    import httpx
    queries = [
        "Docling parser là gì?",
        "Module nào xử lý CSV?",
        "Risk OCR fail có mức severity gì?",
    ]
    async with httpx.AsyncClient(timeout=TIMEOUT_LONG) as client:
        tasks = [client.post(BASE + "/query/ask", json={
            "owner_id": OWNER, "collection_id": COLL_KT, "query": q, "top_k": 3
        }) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r.status_code if hasattr(r, 'status_code') else f"exc: {type(r).__name__}" for r in results]
try:
    statuses = asyncio.run(concurrent_calls())
    ok = all(s == 200 for s in statuses)
    step("3.1_concurrent", ok, f"statuses={statuses} t={time.time()-t:.1f}s")
except Exception as e:
    step("3.1_concurrent", False, f"exc={e}")

# 3.2 Long conversation (5 turns)
t = time.time()
try:
    conv_id = f"e2e_long_{uuid.uuid4().hex[:6]}"
    turns = [
        "Docling parser là gì?",
        "Nó xử lý loại file nào?",
        "Còn module nào khác trong test deck?",
        "Risk OCR fail có severity gì?",
        "Mitigation cho risk đó là gì?",
    ]
    all_pass = True
    for i, q in enumerate(turns):
        r = post("/query/ask", {"owner_id": OWNER, "collection_id": COLL_KT, "query": q,
                                 "conversation_id": conv_id, "top_k": 3}, timeout=TIMEOUT_LONG)
        if r.status_code != 200:
            all_pass = False
            break
    step("3.2_long_conversation", all_pass, f"5 turns conv_id={conv_id} t={time.time()-t:.1f}s")
except Exception as e:
    step("3.2_long_conversation", False, f"exc={e}")

# 3.3 Empty collection (create + query without uploads)
t = time.time()
try:
    rc = post("/collections", {"owner_id": E2E_OWNER, "name": "Empty test"})
    if rc.status_code in (200, 201):
        empty_id = rc.json()["data"]["collection_id"]
        rq = post("/query/ask", {"owner_id": E2E_OWNER, "collection_id": empty_id, "query": "What is this?", "top_k": 3}, timeout=TIMEOUT_LONG)
        d = rq.json().get("data") or {}
        refused = bool(d.get("was_refused"))
        step("3.3_empty_collection", refused, f"http={rq.status_code} refused={refused} t={time.time()-t:.1f}s")
        # Cleanup
        requests.delete(BASE + f"/collections/{empty_id}", params={"owner_id": E2E_OWNER}, timeout=10)
    else:
        step("3.3_empty_collection", False, f"collection create http={rc.status_code}")
except Exception as e:
    step("3.3_empty_collection", False, f"exc={e}")

# 3.4 Cross-lingual EN query on mostly-VN docs
t = time.time()
try:
    p = {"owner_id": OWNER, "collection_id": COLL_TK, "query": "Why was WAPE chosen instead of MAPE?", "top_k": 5, "answer_language": "en"}
    r = post("/query/ask", p, timeout=TIMEOUT_LONG)
    if r.status_code == 200:
        d = r.json().get("data") or {}
        ans = d.get("answer", "") or ""
        ok = bool(ans.strip()) and len(d.get("citations", [])) > 0 and not d.get("was_refused")
        save("3.4_cross_lingual", r.json())
        step("3.4_cross_lingual", ok, f"refused={d.get('was_refused')} lang={d.get('answer_language')} cites={len(d.get('citations',[]))} t={time.time()-t:.1f}s | {trunc(ans, 150)}")
    else:
        step("3.4_cross_lingual", False, f"http={r.status_code}")
except Exception as e:
    step("3.4_cross_lingual", False, f"exc={e}")

# 3.5 Reranker ablation
t = time.time()
try:
    p = {"owner_id": OWNER, "collection_id": COLL_KT, "query": "Module Docling parser là gì?", "top_k": 3,
         "rag_flags": {"reranker_enabled": False}}
    r = post("/query/ask", p, timeout=TIMEOUT_LONG)
    if r.status_code == 200:
        d = r.json().get("data") or {}
        step("3.5_reranker_ablation", not d.get("was_refused"), f"refused={d.get('was_refused')} cites={len(d.get('citations',[]))} t={time.time()-t:.1f}s")
    else:
        step("3.5_reranker_ablation", False, f"http={r.status_code} body={trunc(r.text)}")
except Exception as e:
    step("3.5_reranker_ablation", False, f"exc={e}")


# ────────────────────────────────────────────────────────────────────────
# Phase 4 — Defensive (5 tests)
# ────────────────────────────────────────────────────────────────────────
print("\n=== Phase 4: Defensive ===")

# 4.1 Path traversal in upload filename
try:
    if coll_id is None:
        rc = post("/collections", {"owner_id": E2E_OWNER, "name": "PathTraversal"})
        coll_id_4 = rc.json()["data"]["collection_id"] if rc.status_code in (200, 201) else None
    else:
        coll_id_4 = coll_id
    with DOC_PATH.open("rb") as fh:
        metadata = json.dumps({"owner_id": E2E_OWNER, "collection_id": coll_id_4, "language": "vi"})
        r = requests.post(BASE + "/materials/upload", data={"metadata": metadata},
                          files={"file": ("../../etc/passwd.pdf", fh, "application/pdf")}, timeout=30)
    # System should either: accept with sanitized filename OR reject with 400
    if r.status_code in (200, 201):
        mat_id = r.json().get("data", {}).get("material_id")
        # Check that mat_id doesn't contain dangerous chars
        step("4.1_path_traversal", bool(mat_id) and ".." not in str(mat_id), f"sanitized mat_id={mat_id}")
    else:
        step("4.1_path_traversal", r.status_code == 400, f"rejected http={r.status_code}")
except Exception as e:
    step("4.1_path_traversal", False, f"exc={e}")

# 4.2 MIME mismatch (text content with .pdf extension)
try:
    fake = b"this is not a pdf"
    metadata = json.dumps({"owner_id": E2E_OWNER, "collection_id": coll_id, "language": "vi"})
    r = requests.post(BASE + "/materials/upload", data={"metadata": metadata},
                      files={"file": ("fake.pdf", fake, "application/pdf")}, timeout=30)
    step("4.2_mime_mismatch", r.status_code == 400, f"http={r.status_code} body={trunc(r.text)}")
except Exception as e:
    step("4.2_mime_mismatch", False, f"exc={e}")

# 4.3 Oversize file (25MB > 20MB limit)
try:
    big = b"%PDF-1.4\n" + b"X" * (25 * 1024 * 1024)
    metadata = json.dumps({"owner_id": E2E_OWNER, "collection_id": coll_id, "language": "vi"})
    r = requests.post(BASE + "/materials/upload", data={"metadata": metadata},
                      files={"file": ("big.pdf", big, "application/pdf")}, timeout=60)
    step("4.3_oversize", r.status_code == 400, f"http={r.status_code} body={trunc(r.text)}")
except Exception as e:
    step("4.3_oversize", False, f"exc={e}")

# 4.4 SQL injection in query
try:
    p = {"owner_id": OWNER, "collection_id": COLL_KT, "query": "'; DROP TABLE materials; --", "top_k": 3}
    r = post("/query/ask", p, timeout=TIMEOUT_LONG)
    # System should respond (not crash); also collection should still exist
    rc = get("/materials", params={"owner_id": OWNER})
    still_alive = rc.status_code == 200 and len(rc.json().get("data", [])) > 0
    step("4.4_sql_injection", r.status_code == 200 and still_alive, f"query http={r.status_code} materials_alive={still_alive}")
except Exception as e:
    step("4.4_sql_injection", False, f"exc={e}")

# 4.5 Oversized query (>4000 chars schema max)
try:
    big_q = "a" * 5000
    p = {"owner_id": OWNER, "collection_id": COLL_KT, "query": big_q, "top_k": 3}
    r = post("/query/ask", p, timeout=15)
    step("4.5_oversized_query", r.status_code in (400, 422), f"http={r.status_code}")
except Exception as e:
    step("4.5_oversized_query", False, f"exc={e}")


# Cleanup
print("\n=== Cleanup ===")
if coll_id:
    requests.delete(BASE + f"/collections/{coll_id}", params={"owner_id": E2E_OWNER}, timeout=15)
# Cleanup any e2e_v3_user collections leftover
for c in get("/collections", params={"owner_id": E2E_OWNER}).json().get("data", []):
    requests.delete(BASE + f"/collections/{c.get('collection_id')}", params={"owner_id": E2E_OWNER}, timeout=15)


# ────────────────────────────────────────────────────────────────────────
print("\n\n=== FINAL ===")
passed = sum(1 for ok, _ in results.values() if ok)
total = len(results)
print(f"Passed: {passed}/{total}")
for name, (ok, detail) in results.items():
    tag = "OK " if ok else "X  "
    print(f"  {tag} {name}: {detail}")
sys.exit(0 if passed == total else 1)
