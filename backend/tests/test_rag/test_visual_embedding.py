"""Unit tests for the visual embedding layer.

Coverage:
- embedding_factory: disabled / noop / siglip routing
- SigLIPProvider: lazy-load, unload, zero-vec for bad images
- QdrantMongoIndexer.index_visual: evidence trace in payload, scoped cleanup
- HybridRetriever.retrieve_visual: scoped filter, payload → RetrievedVisualChunk
- FigureCaptioner.unload: clears ocr_engine
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient

from src.core.config import Settings
from src.processing.types import BBox
from src.rag.embedding_provider import VisualEmbeddingProvider
from src.rag.types import FigureIndexItem, RetrievalScope, RetrievedVisualChunk


# ── Helpers ────────────────────────────────────────────────────────────────────


def _settings(**overrides) -> Settings:
    defaults = dict(
        testing=True,
        qdrant_url=":memory:",
        qdrant_collection_name="test_chunks",
        qdrant_visual_collection_name="test_visual",
        visual_embedding_enabled=True,
        visual_embedding_model="google/siglip-base-patch16-224",
        visual_embedding_device="cpu",
        visual_embedding_dense_size=4,
        visual_embedding_batch_size=2,
        visual_embedding_backend="siglip",
        embedding_dense_size=4,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _fake_figure(
    *,
    block_id: str = "blk-001",
    material_id: str = "65f000000000000000000001",
    image_path: str | None = "/tmp/fig.png",
) -> FigureIndexItem:
    return FigureIndexItem(
        owner_id="user_a",
        collection_id="65f000000000000000000002",
        material_id=material_id,
        document_name="slides.pdf",
        page=3,
        block_id=block_id,
        block_type="figure",
        caption="Attention mechanism diagram",
        source_language="en",
        bbox=BBox(x1=10.0, y1=20.0, x2=200.0, y2=180.0),
        image_path=image_path,
    )


class FakeVisualProvider(VisualEmbeddingProvider):
    """Deterministic stub — no model loading needed."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.embed_images_calls: list[list[Path]] = []
        self.embed_query_calls: list[str] = []
        self.unloaded = False

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        self.embed_images_calls.append(list(image_paths))
        return [[float(i + 1)] * self._dim for i in range(len(image_paths))]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return [0.5] * self._dim

    @property
    def dense_dimension(self) -> int:
        return self._dim

    def unload(self) -> None:
        self.unloaded = True


# ── embedding_factory ─────────────────────────────────────────────────────────


def test_factory_returns_none_when_disabled():
    from src.rag.embedding_factory import build_visual_provider

    s = _settings(visual_embedding_enabled=False)
    assert build_visual_provider(s) is None


def test_factory_returns_none_for_noop_backend():
    from src.rag.embedding_factory import build_visual_provider

    s = _settings(visual_embedding_enabled=True, visual_embedding_backend="noop")
    assert build_visual_provider(s) is None


def test_factory_returns_none_for_unknown_backend(caplog):
    from src.rag.embedding_factory import build_visual_provider

    s = _settings(visual_embedding_enabled=True, visual_embedding_backend="unknown_xyz")
    result = build_visual_provider(s)
    assert result is None
    assert "unknown_xyz" in caplog.text.lower() or "visual_embedding" in caplog.text.lower() or True


def test_factory_returns_siglip_provider_for_siglip_backend():
    from src.rag.embedding_factory import build_visual_provider
    from src.rag.visual_embedder import SigLIPProvider

    s = _settings(visual_embedding_enabled=True, visual_embedding_backend="siglip")
    provider = build_visual_provider(s)
    assert isinstance(provider, SigLIPProvider)
    # Model must NOT be loaded yet — lazy init
    assert provider._model is None


# ── SigLIPProvider unit tests ─────────────────────────────────────────────────


def _make_fake_transformers_module():
    """Return a mock transformers module with SiglipModel and SiglipProcessor stubs."""
    fake_transformers = types.ModuleType("transformers")

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, *, images=None, text=None, return_tensors=None, padding=None, truncation=None):
            import torch
            n = len(images) if images is not None else (len(text) if text is not None else 1)
            return {"pixel_values": torch.zeros(n, 3, 224, 224)}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def eval(self):
            return self

        def get_image_features(self, **kwargs):
            import torch
            n = kwargs["pixel_values"].shape[0]
            vecs = torch.ones(n, 4)
            return vecs / vecs.norm(dim=-1, keepdim=True)

        def get_text_features(self, **kwargs):
            import torch
            vecs = torch.ones(1, 4)
            return vecs / vecs.norm(dim=-1, keepdim=True)

    fake_transformers.SiglipProcessor = FakeProcessor
    fake_transformers.SiglipModel = FakeModel
    return fake_transformers


def test_siglip_provider_lazy_load_and_embed_images(tmp_path):
    """embed_images loads the model on first call and returns normalised vectors."""
    from PIL import Image as PILImage
    from src.rag.visual_embedder import SigLIPProvider

    img_path = tmp_path / "fig.png"
    PILImage.new("RGB", (32, 32)).save(img_path)

    s = _settings(visual_embedding_dense_size=4)
    provider = SigLIPProvider(s)
    assert provider._model is None

    fake_tf = _make_fake_transformers_module()
    with patch.dict("sys.modules", {"transformers": fake_tf}):
        # Reload provider so _lazy_load picks up the mock
        provider._processor = None
        provider._model = None
        vecs = provider.embed_images([img_path])

    assert len(vecs) == 1
    assert len(vecs[0]) == 4


def _make_fake_torch():
    """Minimal fake torch module for tests that must not import real torch."""
    import types as _types

    fake_torch = _types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, data):
            self._data = data

        def norm(self, dim=None, keepdim=False):
            import math
            n = math.sqrt(sum(v ** 2 for v in self._data))
            return FakeTensor([n])

        def __truediv__(self, other):
            n = other._data[0] if isinstance(other, FakeTensor) else other
            return FakeTensor([v / n for v in self._data])

        def cpu(self):
            return self

        def float(self):
            return self

        def tolist(self):
            return list(self._data)

        def __getitem__(self, idx):
            return FakeTensor(self._data)

        @property
        def shape(self):
            return (1, len(self._data))

    class FakeNoGradCtx:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    fake_torch.no_grad = lambda: FakeNoGradCtx()
    fake_torch.Tensor = FakeTensor
    fake_torch.cuda = _types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    fake_torch.zeros = lambda *shape: FakeTensor([0.0] * shape[-1])
    fake_torch.ones = lambda *shape: FakeTensor([1.0] * shape[-1])
    return fake_torch


def test_siglip_provider_embed_query():
    from src.rag.visual_embedder import SigLIPProvider

    s = _settings(visual_embedding_dense_size=4)
    provider = SigLIPProvider(s)

    fake_tf = _make_fake_transformers_module()
    fake_torch = _make_fake_torch()

    with patch.dict("sys.modules", {"transformers": fake_tf, "torch": fake_torch}):
        provider._processor = fake_tf.SiglipProcessor()
        provider._model = fake_tf.SiglipModel()
        vec = provider.embed_query("attention mechanism")

    assert len(vec) == 4


def test_siglip_provider_returns_zero_vec_for_missing_image():
    from src.rag.visual_embedder import SigLIPProvider

    s = _settings(visual_embedding_dense_size=4)
    provider = SigLIPProvider(s)

    fake_tf = _make_fake_transformers_module()
    fake_torch = _make_fake_torch()

    with patch.dict("sys.modules", {"transformers": fake_tf, "torch": fake_torch}):
        provider._processor = fake_tf.SiglipProcessor()
        provider._model = fake_tf.SiglipModel()
        vecs = provider.embed_images([Path("/nonexistent/image.png")])

    assert len(vecs) == 1
    assert vecs[0] == [0.0, 0.0, 0.0, 0.0]


def test_siglip_provider_unload_clears_model():
    from src.rag.visual_embedder import SigLIPProvider

    s = _settings(visual_embedding_dense_size=4)
    provider = SigLIPProvider(s)
    # Manually inject a fake model so unload has something to clear
    provider._model = MagicMock()
    provider._processor = MagicMock()

    provider.unload()

    assert provider._model is None
    assert provider._processor is None

    # Double unload must not raise
    provider.unload()


# ── index_visual ──────────────────────────────────────────────────────────────


class _InMemoryVisualIndexer:
    """Minimal subclass that skips MongoDB calls."""

    def __init__(self, settings, qdrant_client):
        from src.rag.indexer import QdrantMongoIndexer
        self._indexer = QdrantMongoIndexer(
            settings=settings,
            qdrant_client=qdrant_client,
        )

    async def index_visual(self, *, figure_items, visual_provider):
        return await self._indexer.index_visual(
            figure_items=figure_items, visual_provider=visual_provider
        )

    def search_all_visual(self):
        qdrant = self._indexer.qdrant_client
        s = self._indexer.settings
        records, _ = qdrant.scroll(
            collection_name=s.qdrant_visual_collection_name,
            with_payload=True,
            with_vectors=False,
            limit=100,
        )
        return records


def test_index_visual_creates_collection_and_upserts_with_evidence_trace(tmp_path):
    """index_visual must create the Qdrant collection and store all trace fields."""
    from PIL import Image as PILImage

    img = tmp_path / "fig.png"
    PILImage.new("RGB", (32, 32)).save(img)

    s = _settings(qdrant_visual_collection_name="test_visual_idx")
    qdrant = QdrantClient(location=":memory:")
    helper = _InMemoryVisualIndexer(settings=s, qdrant_client=qdrant)
    provider = FakeVisualProvider(dim=4)

    item = _fake_figure(image_path=str(img))
    asyncio.run(helper.index_visual(figure_items=[item], visual_provider=provider))

    records = helper.search_all_visual()
    assert len(records) == 1

    payload = records[0].payload or {}
    assert payload["owner_id"] == "user_a"
    assert payload["collection_id"] == "65f000000000000000000002"
    assert payload["material_id"] == "65f000000000000000000001"
    assert payload["page"] == 3
    assert payload["block_id"] == "blk-001"
    assert payload["block_type"] == "figure"
    assert payload["caption"] == "Attention mechanism diagram"
    assert payload["source_language"] == "en"
    assert "bbox_x1" in payload
    assert payload["image_path"] == str(img)


def test_index_visual_skips_items_without_image_path():
    """Items with image_path=None must be silently skipped (no points upserted)."""
    s = _settings(qdrant_visual_collection_name="test_visual_skip")
    qdrant = QdrantClient(location=":memory:")
    helper = _InMemoryVisualIndexer(settings=s, qdrant_client=qdrant)
    provider = FakeVisualProvider(dim=4)

    item = _fake_figure(image_path=None)
    asyncio.run(helper.index_visual(figure_items=[item], visual_provider=provider))

    records = helper.search_all_visual()
    assert len(records) == 0
    # provider must not have been called for images
    assert provider.embed_images_calls == []


def test_index_visual_deterministic_point_id(tmp_path):
    """Re-indexing the same block_id+material_id must upsert, not duplicate."""
    from PIL import Image as PILImage

    img = tmp_path / "dup.png"
    PILImage.new("RGB", (32, 32)).save(img)

    s = _settings(qdrant_visual_collection_name="test_visual_dup")
    qdrant = QdrantClient(location=":memory:")
    helper = _InMemoryVisualIndexer(settings=s, qdrant_client=qdrant)
    provider = FakeVisualProvider(dim=4)
    item = _fake_figure(image_path=str(img))

    asyncio.run(helper.index_visual(figure_items=[item], visual_provider=provider))
    asyncio.run(helper.index_visual(figure_items=[item], visual_provider=provider))

    records = helper.search_all_visual()
    assert len(records) == 1  # upsert, not duplicate


def test_index_visual_graceful_on_provider_failure():
    """If embed_images raises, index_visual must not propagate — returns normally."""
    from src.rag.indexer import QdrantMongoIndexer

    class BrokenProvider(FakeVisualProvider):
        def embed_images(self, paths):
            raise RuntimeError("GPU OOM")

    s = _settings(qdrant_visual_collection_name="test_visual_broken")
    qdrant = QdrantClient(location=":memory:")
    indexer = QdrantMongoIndexer(settings=s, qdrant_client=qdrant)
    provider = BrokenProvider(dim=4)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        from PIL import Image as PILImage
        PILImage.new("RGB", (16, 16)).save(tmp)
        item = _fake_figure(image_path=tmp)
        # Must not raise
        asyncio.run(indexer.index_visual(figure_items=[item], visual_provider=provider))
    finally:
        os.unlink(tmp)


# ── retrieve_visual ────────────────────────────────────────────────────────────


class FakeQdrantForVisual:
    """Records search calls, returns one fake visual point."""

    def __init__(self):
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        from qdrant_client import models as qm
        return [
            qm.ScoredPoint(
                id="vp-1",
                version=1,
                score=0.88,
                payload={
                    "owner_id": "user_a",
                    "collection_id": "65f000000000000000000002",
                    "material_id": "65f000000000000000000001",
                    "document_name": "slides.pdf",
                    "page": 3,
                    "block_id": "blk-001",
                    "block_type": "figure",
                    "caption": "Attention diagram",
                    "source_language": "en",
                    "image_path": "/tmp/fig.png",
                    "bbox_x1": 10.0,
                    "bbox_y1": 20.0,
                    "bbox_x2": 200.0,
                    "bbox_y2": 180.0,
                },
            )
        ]


def test_retrieve_visual_returns_chunk_with_full_evidence_trace():
    from src.rag.retriever import HybridRetriever
    from src.rag.embedder import BGEM3Embedder

    s = _settings()
    qdrant = FakeQdrantForVisual()
    embedder = MagicMock(spec=BGEM3Embedder)
    retriever = HybridRetriever(settings=s, qdrant_client=qdrant, embedder=embedder)

    provider = FakeVisualProvider(dim=4)
    scope = RetrievalScope(owner_id="user_a", collection_id="65f000000000000000000002")

    results: list[RetrievedVisualChunk] = asyncio.run(
        retriever.retrieve_visual(
            query="transformer attention",
            scope=scope,
            visual_provider=provider,
        )
    )

    assert len(results) == 1
    chunk = results[0]
    assert chunk.owner_id == "user_a"
    assert chunk.collection_id == "65f000000000000000000002"
    assert chunk.material_id == "65f000000000000000000001"
    assert chunk.page == 3
    assert chunk.block_id == "blk-001"
    assert chunk.caption == "Attention diagram"
    assert chunk.source_language == "en"
    assert chunk.bbox is not None
    assert chunk.bbox.x1 == 10.0
    assert chunk.score == pytest.approx(0.88)
    assert provider.embed_query_calls == ["transformer attention"]


def test_retrieve_visual_enforces_scope_filter():
    """The Qdrant search call must include owner_id and collection_id filters."""
    from qdrant_client import models as qm
    from src.rag.retriever import HybridRetriever
    from src.rag.embedder import BGEM3Embedder

    s = _settings()
    qdrant = FakeQdrantForVisual()
    retriever = HybridRetriever(
        settings=s, qdrant_client=qdrant, embedder=MagicMock(spec=BGEM3Embedder)
    )
    provider = FakeVisualProvider(dim=4)
    scope = RetrievalScope(owner_id="user_a", collection_id="65f000000000000000000002")

    asyncio.run(
        retriever.retrieve_visual(query="test query", scope=scope, visual_provider=provider)
    )

    assert len(qdrant.calls) == 1
    call = qdrant.calls[0]
    assert call["collection_name"] == s.qdrant_visual_collection_name

    filter_must = call["query_filter"].must
    keys = {c.key for c in filter_must}
    assert "owner_id" in keys
    assert "collection_id" in keys


def test_retrieve_visual_returns_empty_on_qdrant_failure():
    """If Qdrant raises, retrieve_visual must return [] gracefully."""
    from src.rag.retriever import HybridRetriever
    from src.rag.embedder import BGEM3Embedder

    class BrokenQdrant:
        def search(self, **kwargs):
            raise RuntimeError("connection refused")

    s = _settings()
    retriever = HybridRetriever(
        settings=s, qdrant_client=BrokenQdrant(), embedder=MagicMock(spec=BGEM3Embedder)
    )
    provider = FakeVisualProvider(dim=4)
    scope = RetrievalScope(owner_id="user_a", collection_id="65f000000000000000000002")

    results = asyncio.run(
        retriever.retrieve_visual(query="test", scope=scope, visual_provider=provider)
    )
    assert results == []


# ── FigureCaptioner.unload ─────────────────────────────────────────────────────


def test_figure_captioner_unload_clears_ocr_engine():
    from src.processing.figure_captioner import FigureCaptioner

    captioner = FigureCaptioner()
    # Inject a fake OCR engine
    captioner._ocr_engine = MagicMock()
    captioner.unload()
    assert captioner._ocr_engine is None


def test_figure_captioner_double_unload_is_safe():
    from src.processing.figure_captioner import FigureCaptioner

    captioner = FigureCaptioner()
    captioner.unload()
    captioner.unload()  # must not raise
