# -*- coding: utf-8 -*-
"""Tests for modality-aware routing + table-native serialization."""
from types import SimpleNamespace

import pytest

from src.inference.inference_engine import InferenceEngine
from src.processing import table_serializer as ts
from src.rag.query_router import (
    QueryRouter, RouteType, PreferredModality, Difficulty, TableQueryType,
)
from src.rag.types import RetrievedChunk


def _chunk(content: str, modality: str = "table") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c-{hash(content) & 0xfff}",
        owner_id="o", collection_id="c", material_id="m",
        document_name="d.pdf", content=content, language="vi", modality=modality,
    )


# ── table_serializer ──────────────────────────────────────────────────────────

class TestTableSerializer:
    def test_to_html_grid(self):
        html = ts.to_html(["Tên", "Điểm"], [["An", "9"], ["Bình", "8"]])
        assert html.startswith("<table><tr><th>Tên</th><th>Điểm</th></tr>")
        assert "<td>An</td><td>9</td>" in html
        assert html.endswith("</table>")

    def test_to_html_escapes_and_pads(self):
        html = ts.to_html(["A", "B"], [["<x>", ""], ["only-one"]])
        assert "&lt;x&gt;" in html          # escaped
        assert "<td></td>" in html          # padded missing cell
        assert html.count("<tr>") == 3      # header + 2 rows

    def test_verbalize_rows_format(self):
        rows = ts.verbalize_rows(["Tên", "Điểm"], [["An", "9"]], table_name="Bảng A", language="vi")
        assert rows == [(1, "Hàng 1 của bảng 'Bảng A'. Tên: An. Điểm: 9.")]

    def test_verbalize_rows_en_and_skip_empty(self):
        rows = ts.verbalize_rows(
            ["A", "B"], [["", ""], ["x", "y"]], table_name="T", language="en", start_index=5
        )
        # first row fully empty → skipped; second keeps its index
        assert rows == [(6, "Row 6 of table 'T'. A: x. B: y.")]

    def test_structured_meta(self):
        meta = ts.structured_meta(["Tên", ""], [["An", "9"]])
        assert meta == {"columns": ["Tên", "Column 2"], "n_rows": 1, "n_cols": 2}

    def test_parse_markdown_table_roundtrip(self):
        md = "| Tên | Điểm |\n| --- | --- |\n| An | 9 |\n| Bình | 8 |"
        assert ts.parse_markdown_table(md) == (["Tên", "Điểm"], [["An", "9"], ["Bình", "8"]])

    def test_parse_markdown_table_rejects_non_table(self):
        assert ts.parse_markdown_table("just a sentence") is None
        assert ts.parse_markdown_table("") is None


# ── modality router ───────────────────────────────────────────────────────────

class TestModalityRouter:
    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_table_query(self, router):
        d = router.route("Giá trị ở cột Điểm của bảng là bao nhiêu?")
        assert d.preferred_modality == PreferredModality.TABLE

    def test_figure_query(self, router):
        d = router.route("Biểu đồ nào thể hiện xu hướng tăng?")
        assert d.preferred_modality == PreferredModality.FIGURE

    def test_audio_query(self, router):
        d = router.route("Ở phút 5 đoạn ghi âm nói gì?")
        assert d.preferred_modality == PreferredModality.AUDIO

    def test_plain_text_query_is_none(self, router):
        d = router.route("Machine learning là gì?")
        assert d.preferred_modality == PreferredModality.NONE

    def test_modality_is_orthogonal_to_route(self, router):
        # comparison over tables: COMPARISON route + TABLE modality
        d = router.route("So sánh giá trị hai bảng số liệu")
        assert d.route_type == RouteType.COMPARISON
        assert d.preferred_modality == PreferredModality.TABLE

    def test_multipliers_unchanged(self, router):
        # guards the limit-scaling behaviour — modality must not alter multipliers
        assert router.route("Machine learning là gì?").top_k_multiplier == 0.75
        assert router.route("Tóm tắt nội dung").top_k_multiplier == 2.0
        assert router.route("X ảnh hưởng thế nào đến Y?").top_k_multiplier == 1.5


# ── structured product-router signals (Phase 2) ───────────────────────────────

class TestStructuredRouter:
    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_table_aggregation_subtype(self, router):
        d = router.route("Sản phẩm nào có giá cao nhất trong bảng?")
        assert d.preferred_modality == PreferredModality.TABLE
        assert d.table_query_type == TableQueryType.AGGREGATION

    def test_table_lookup_subtype(self, router):
        d = router.route("Giá của iPhone trong bảng là bao nhiêu?")
        assert d.preferred_modality == PreferredModality.TABLE
        assert d.table_query_type == TableQueryType.LOOKUP

    def test_non_table_has_no_subtype(self, router):
        d = router.route("Machine learning là gì?")
        assert d.table_query_type is None

    def test_simple_query_high_confidence_no_agentic(self, router):
        d = router.route("Machine learning là gì?")
        assert d.difficulty == Difficulty.SIMPLE
        assert d.confidence >= 0.8
        assert d.should_use_agentic is False

    def test_complex_query_triggers_agentic(self, router):
        d = router.route("So sánh các tài liệu theo thời gian và mối quan hệ giữa chúng")
        assert d.difficulty == Difficulty.COMPLEX
        assert d.should_use_agentic is True

    def test_graph_relation_is_multi_hop(self, router):
        d = router.route("X ảnh hưởng thế nào đến Y?")
        assert d.difficulty in (Difficulty.MULTI_HOP, Difficulty.COMPLEX)


# ── conditional table drop in _filter_substantive_chunks ──────────────────────

class TestSubstantiveChunkFilter:
    def _fake_self(self):
        return SimpleNamespace(
            settings=SimpleNamespace(
                inference_substantive_chunk_filter_prefixes=["Nguồn:"],
                inference_min_chunk_chars=40,
            )
        )

    def test_table_route_keeps_rows_and_grid(self):
        chunks = [
            _chunk("Hàng 3 của bảng 'Điểm'. Tên: An. Điểm: 9. Xếp loại: Giỏi rất tốt."),
            _chunk("<table><tr><th>Tên</th><th>Điểm</th></tr><tr><td>An</td><td>9</td></tr></table>"),
        ]
        kept = InferenceEngine._filter_substantive_chunks(
            self._fake_self(), chunks, preferred_modality="table"
        )
        assert len(kept) == 2  # both survive on the table route

    def test_non_table_route_drops_rows_and_grid(self):
        chunks = [
            _chunk("Hàng 3 của bảng 'Điểm'. Tên: An. Điểm: 9. Xếp loại: Giỏi rất tốt."),
            _chunk("<table><tr><th>Tên</th></tr><tr><td>An</td></tr></table>"),
            _chunk("| Tên | Điểm |\n| --- | --- |\n| An | 9 |"),
            _chunk("Dropout giảm overfitting bằng cách tắt ngẫu nhiên các nơ-ron trong huấn luyện.", modality="text"),
        ]
        kept = InferenceEngine._filter_substantive_chunks(
            self._fake_self(), chunks, preferred_modality=None
        )
        # rows + html grid + markdown grid dropped; only the prose paragraph stays
        assert len(kept) == 1
        assert kept[0].modality == "text"

    def test_en_rows_also_dropped_on_non_table_route(self):
        # fixes the latent VI-only asymmetry
        chunks = [
            _chunk("Row 2 of table 'Scores'. Name: An. Score: 9. Grade: Excellent overall."),
            _chunk("Dropout giảm overfitting bằng cách tắt ngẫu nhiên các nơ-ron trong huấn luyện.", modality="text"),
        ]
        kept = InferenceEngine._filter_substantive_chunks(
            self._fake_self(), chunks, preferred_modality=None
        )
        assert len(kept) == 1 and kept[0].modality == "text"
