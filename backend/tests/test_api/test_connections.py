from __future__ import annotations

import asyncio

from src.core.config import Settings
from src.database import DOCUMENT_MODELS, init_database
from src.rag.vector_store import build_qdrant_client


def test_qdrant_client_can_use_in_memory_backend() -> None:
    client = build_qdrant_client(Settings(testing=True, qdrant_url=":memory:"))
    assert client.get_collections().collections == []


def test_database_initialization_calls_beanie(monkeypatch) -> None:
    calls = {}

    class FakeMotorClient:
        def __init__(self, uri: str) -> None:
            calls["uri"] = uri

        def __getitem__(self, database_name: str) -> str:
            calls["database_name"] = database_name
            return f"db:{database_name}"

    async def fake_init_beanie(*, database, document_models) -> None:
        calls["database"] = database
        calls["document_models"] = document_models

    monkeypatch.setattr("src.database.AsyncIOMotorClient", FakeMotorClient)
    monkeypatch.setattr("src.database.init_beanie", fake_init_beanie)

    settings = Settings(
        testing=False,
        mongodb_uri="mongodb+srv://example.invalid/agentbook",
        mongodb_database="agentbook_test",
    )
    asyncio.run(init_database(settings))

    assert calls["uri"] == "mongodb+srv://example.invalid/agentbook"
    assert calls["database"] == "db:agentbook_test"
    assert calls["document_models"] == DOCUMENT_MODELS
