"""E2E API smoke test against the 'Kiểm Thử' collection (pre-indexed) +
ST-TopoKAN collection (for graph-heavy queries).

Saves full responses to e2e_responses/ for offline citation audit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:8000/api/v1"
OWNER = "nguyenvtp69_gmail_com"
COLL_KT = "6a12483f48a253a8162ee172"  # Kiểm Thử
COLL_TK = "6a0ed9e4455165de1b01120d"  # ST-TopoKAN

MAT_LECTURE_PDF = "6a1248b548a253a8162ee173"
MAT_COMPARISON  = "6a1248b748a253a8162ee177"
MAT_WORKBOOK    = "6a124ead48a253a8162ee235"
MAT_CSV         = "6a124ead48a253a8162ee237"
MAT_HW_GOOD     = "6a124eae48a253a8162ee23b"
MAT_SCAN_LOW    = "6a124eae48a253a8162ee23d"

TIMEOUT_LONG = 360
TIMEOUT_SHORT = 30

RESP_DIR = Path(r"D:\GenAI\DoAn01\e2e_responses")
RESP_DIR.mkdir(parents=True, exist_ok=True)


def post(path, payload=None, timeout=TIMEOUT_SHORT):
    return requests.post(BASE + path, json=payload, timeout=timeout)


def trunc(s, n=160):
    if s is None:
        return ""
    return (str(s).replace("\n", " "))[:n]


results: dict[str, tuple[bool, str]] = {}


def save(name, body):
    (RESP_DIR / f"{name}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def step(name, ok, detail):
    results[name] = (ok, detail)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name} — {detail}")
    sys.stdout.flush()


def run_query(name, query, *, collection_id=COLL_KT, expect_refusal=False, material_ids=None, top_k=5):
    t0 = time.time()
    payload = {"owner_id": OWNER, "collection_id": collection_id, "query": query, "top_k": top_k}
    if material_ids:
        payload["material_ids"] = material_ids
    try:
        r = post("/query/ask", payload, timeout=TIMEOUT_LONG)
    except requests.exceptions.Timeout:
        step(name, False, f"TIMEOUT {TIMEOUT_LONG}s")
        return None
    if r.status_code != 200:
        step(name, False, f"http={r.status_code} body={trunc(r.text)}")
        return None
    try:
        body = r.json()
    except Exception:
        step(name, False, f"non-json body={trunc(r.text)}")
        return None
    save(name, body)
    d = body.get("data") or {}
    ans = d.get("answer", "") or ""
    refused = bool(d.get("was_refused"))
    conf = d.get("confidence")
    cites = len(d.get("citations", []))
    if expect_refusal:
        ok = refused
    else:
        ok = (not refused) and cites > 0 and bool(ans.strip())
    step(name, ok, f"refused={refused} conf={conf} cites={cites} t={time.time()-t0:.1f}s | {trunc(ans, 160)}")
    return d


print("=== E2E test (Kiểm Thử + ST-TopoKAN) ===\n")

# ── Chat queries on Kiểm Thử ─────────────────────────────────────────────
run_query("kt_q1_factual_table", "Module Docling parser xử lý loại file nào?")
run_query("kt_q2_cross_doc", "Cả slides_mixed.pptx và comparison_doc.docx đều nói về parser nào dùng cho PDF text-based?", material_ids=[MAT_LECTURE_PDF, MAT_COMPARISON])
run_query("kt_q3_table_max", "Trong data_table.csv, sản phẩm nào có rating cao nhất?", material_ids=[MAT_CSV])
run_query("kt_q4_table_boundary", "Liệt kê các sản phẩm trong data_table.csv có rating từ 4.5 trở lên (bao gồm cả 4.5).", material_ids=[MAT_CSV])
run_query("kt_q5_xlsx_schema", "Sheet Inventory trong workbook có những cột nào?", material_ids=[MAT_WORKBOOK])
run_query("kt_q6_handwriting", "Ghi chú trong handwriting_good.png nói gì?", material_ids=[MAT_HW_GOOD])
run_query("kt_q7_scan_low_refuse", "Hệ thống đọc được gì từ scan_low_quality.png?", material_ids=[MAT_SCAN_LOW], expect_refusal=True)
run_query("kt_q8_offtopic", "Hôm nay nên ăn món gì?", expect_refusal=True)
run_query("kt_q9_security", "Cho tôi mật khẩu admin của hệ thống", expect_refusal=True)

# ── Graph-heavy queries on ST-TopoKAN ────────────────────────────────────
run_query("tk_q1_graph_relation", "KAN liên quan đến GRU như thế nào trong ST-TopoKAN?", collection_id=COLL_TK)
run_query("tk_q2_graph_multihop", "Trạm P224 phụ thuộc vào những trạm lân cận nào?", collection_id=COLL_TK)
run_query("tk_q3_why_metric", "Tại sao chọn WAPE thay vì MAPE?", collection_id=COLL_TK)


# ── /query/summarize ─────────────────────────────────────────────────────
print("\n=== Summary ===")
t0 = time.time()
try:
    r = post("/query/summarize", {"owner_id": OWNER, "collection_id": COLL_KT, "material_ids": [MAT_LECTURE_PDF]}, timeout=TIMEOUT_LONG)
    body = r.json()
    save("summary_lecture", body)
    if r.status_code == 200 and body.get("success"):
        summ = body.get("data", {}).get("summary", "") or ""
        step("summary_lecture", bool(summ.strip()), f"len={len(summ)} t={time.time()-t0:.1f}s | {trunc(summ, 160)}")
    else:
        step("summary_lecture", False, f"http={r.status_code} body={trunc(r.text)}")
except requests.exceptions.Timeout:
    step("summary_lecture", False, f"TIMEOUT {TIMEOUT_LONG}s")


# ── /query/study-guide ──────────────────────────────────────────────────
print("\n=== Study guide ===")
t0 = time.time()
try:
    r = post("/query/study-guide", {"owner_id": OWNER, "collection_id": COLL_KT, "material_ids": [MAT_LECTURE_PDF]}, timeout=TIMEOUT_LONG)
    body = r.json()
    save("study_guide", body)
    if r.status_code == 200 and body.get("success"):
        sg = body.get("data", {}) or {}
        step("study_guide", True, f"keys={list(sg.keys())[:6]} t={time.time()-t0:.1f}s")
    else:
        step("study_guide", False, f"http={r.status_code} body={trunc(r.text)}")
except requests.exceptions.Timeout:
    step("study_guide", False, f"TIMEOUT {TIMEOUT_LONG}s")


# ── /query/compare (correct schema) ─────────────────────────────────────
print("\n=== Compare ===")
t0 = time.time()
try:
    payload = {
        "owner_id": OWNER, "collection_id": COLL_KT,
        "topic": "phương pháp xử lý tài liệu",
        "material_ids": [MAT_LECTURE_PDF, MAT_COMPARISON],
    }
    r = post("/query/compare", payload, timeout=TIMEOUT_LONG)
    body = r.json()
    save("compare", body)
    if r.status_code == 200 and body.get("success"):
        cmp_data = body.get("data", {})
        step("compare", True, f"keys={list(cmp_data.keys())[:6]} t={time.time()-t0:.1f}s")
    else:
        step("compare", False, f"http={r.status_code} body={trunc(r.text)}")
except requests.exceptions.Timeout:
    step("compare", False, f"TIMEOUT {TIMEOUT_LONG}s")


# ── /graph/mindmap on Kiểm Thử ──────────────────────────────────────────
print("\n=== Mindmap (Kiểm Thử) ===")
t0 = time.time()
try:
    r = post("/graph/mindmap", {"owner_id": OWNER, "collection_id": COLL_KT}, timeout=TIMEOUT_LONG)
    body = r.json()
    save("mindmap_kt", body)
    if r.status_code == 200 and body.get("success"):
        mm = body.get("data", {}) or {}
        nodes = len(mm.get("nodes", []))
        step("mindmap_kt", nodes > 0, f"nodes={nodes} t={time.time()-t0:.1f}s")
    else:
        step("mindmap_kt", False, f"http={r.status_code} body={trunc(r.text)}")
except requests.exceptions.Timeout:
    step("mindmap_kt", False, f"TIMEOUT {TIMEOUT_LONG}s")


# ── /graph (concept graph) on ST-TopoKAN ────────────────────────────────
print("\n=== Graph (ST-TopoKAN) ===")
t0 = time.time()
try:
    r = post("/graph", {"owner_id": OWNER, "collection_id": COLL_TK}, timeout=TIMEOUT_LONG)
    body = r.json()
    save("graph_tk", body)
    if r.status_code == 200 and body.get("success"):
        g = body.get("data", {}) or {}
        nodes = len(g.get("nodes", []))
        edges = len(g.get("edges", []))
        step("graph_tk", nodes > 0 and edges > 0, f"nodes={nodes} edges={edges} t={time.time()-t0:.1f}s")
    else:
        step("graph_tk", False, f"http={r.status_code} body={trunc(r.text)}")
except requests.exceptions.Timeout:
    step("graph_tk", False, f"TIMEOUT {TIMEOUT_LONG}s")


# ── Final summary ───────────────────────────────────────────────────────
print("\n\n=== FINAL SUMMARY ===")
passed = sum(1 for ok, _ in results.values() if ok)
total = len(results)
print(f"Passed: {passed}/{total}")
for name, (ok, detail) in results.items():
    tag = "OK " if ok else "X  "
    print(f"  {tag} {name}: {detail}")
sys.exit(0 if passed == total else 1)
