# -*- coding: utf-8 -*-
"""Tests for the per-request observability trace (Phase 1)."""
import time

from src.core.trace import RequestTrace
from src.models.query_log import RequestTraceModel


def test_query_id_is_generated_and_stable():
    t = RequestTrace()
    assert t.query_id and len(t.query_id) >= 16
    assert t.to_dict()["query_id"] == t.query_id


def test_explicit_query_id_preserved():
    t = RequestTrace(query_id="abc123")
    assert t.query_id == "abc123"


def test_stage_records_latency():
    t = RequestTrace()
    with t.stage("retrieve"):
        time.sleep(0.01)
    assert "retrieve" in t.latency_by_stage
    assert t.latency_by_stage["retrieve"] >= 5


def test_stage_accumulates_on_reentry():
    t = RequestTrace()
    with t.stage("retrieve"):
        time.sleep(0.005)
    first = t.latency_by_stage["retrieve"]
    with t.stage("retrieve"):
        time.sleep(0.005)
    assert t.latency_by_stage["retrieve"] >= first


def test_set_skips_none_and_to_dict_merges_fields():
    t = RequestTrace()
    t.set("route", "factual")
    t.set("modality", None)          # dropped
    t.update(prompt_file="qa_table.txt", retrieved_chunk_ids=["c1", "c2"])
    d = t.to_dict()
    assert d["route"] == "factual"
    assert "modality" not in d
    assert d["prompt_file"] == "qa_table.txt"
    assert d["retrieved_chunk_ids"] == ["c1", "c2"]
    assert "latency_by_stage" in d


def test_trace_dict_maps_into_log_model():
    t = RequestTrace()
    t.update(route="factual", modality="table", table_query_type="aggregation",
             prompt_file="qa_table.txt", rerank_scores=[0.9, 0.8])
    with t.stage("generate"):
        pass
    data = t.to_dict()
    model = RequestTraceModel(**{k: v for k, v in data.items() if k in RequestTraceModel.model_fields})
    assert model.query_id == t.query_id
    assert model.route == "factual"
    assert model.modality == "table"
    assert model.table_query_type == "aggregation"
    assert model.rerank_scores == [0.9, 0.8]
    assert "generate" in model.latency_by_stage
