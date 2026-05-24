"""Unit tests for MemoryService — all Beanie DB calls are mocked."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.memory_service import MemoryService
from src.models.chat_memory import ChatTurn

MODULE = "src.services.memory_service"


def _service(max_turns: int = 5) -> MemoryService:
    return MemoryService(max_turns=max_turns)


def _turn(role: str, content: str) -> ChatTurn:
    return ChatTurn(role=role, content=content)  # type: ignore[arg-type]


def _make_doc(turns: list[ChatTurn]) -> MagicMock:
    doc = MagicMock()
    doc.turns = list(turns)
    doc.save = AsyncMock()
    doc.delete = AsyncMock()
    doc.insert = AsyncMock()
    return doc


def _patch_find(return_value):
    """Patch ChatMemory.find_one in the service module namespace."""
    mock = AsyncMock(return_value=return_value)
    return patch(f"{MODULE}.ChatMemory.find_one", mock)


# ── get_context ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_context_empty_returns_empty_string():
    """None stored document → empty string, not an error."""
    svc = _service()
    with _patch_find(None):
        result = await svc.get_context("u1:conv1")
    assert result == ""


@pytest.mark.asyncio
async def test_get_context_none_memory_context_query_unchanged():
    """Empty context → anaphora resolution leaves query unchanged."""
    svc = _service()
    with _patch_find(None):
        context = await svc.get_context("u1:conv1")
    assert context == ""
    query = "What is backpropagation?"
    # Simulate what QueryService does: empty context → no prefix appended
    effective = query if not context.strip() else f"...prefix...\nFollow-up: {query}"
    assert effective == query


@pytest.mark.asyncio
async def test_get_context_formats_turns_correctly():
    """Turns must appear as 'User: ...' / 'Assistant: ...' lines."""
    svc = _service()
    doc = _make_doc([
        _turn("user", "What is dropout?"),
        _turn("assistant", "Dropout is a regularization technique."),
    ])
    with _patch_find(doc):
        result = await svc.get_context("u1:conv1")
    assert "User: What is dropout?" in result
    assert "Assistant: Dropout is a regularization technique." in result


@pytest.mark.asyncio
async def test_get_context_respects_last_n_parameter():
    """last_n=1 → only the last exchange (2 turns) in context."""
    svc = _service(max_turns=10)
    turns = [
        _turn("user", "First question"),
        _turn("assistant", "First answer"),
        _turn("user", "Second question"),
        _turn("assistant", "Second answer"),
    ]
    doc = _make_doc(turns)
    with _patch_find(doc):
        result = await svc.get_context("u1:conv1", last_n=1)
    assert "Second question" in result
    assert "First question" not in result


# ── update ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_creates_new_document_when_none_exists():
    """update() on a fresh session must insert a new document."""
    svc = _service()
    mock_doc = MagicMock()
    mock_doc.insert = AsyncMock()

    MockChatMemory = MagicMock()
    MockChatMemory.find_one = AsyncMock(return_value=None)
    MockChatMemory.return_value = mock_doc

    with patch(f"{MODULE}.ChatMemory", MockChatMemory):
        await svc.update("u1:conv1", "Hello", "Hi there", "col1", owner_id="u1")

    mock_doc.insert.assert_called_once()


@pytest.mark.asyncio
async def test_update_appends_turns_to_existing_doc():
    """update() on existing doc appends 2 turns and saves."""
    svc = _service()
    doc = _make_doc([_turn("user", "Old question"), _turn("assistant", "Old answer")])
    with _patch_find(doc):
        await svc.update("u1:conv1", "New question", "New answer", owner_id="u1")
    assert len(doc.turns) == 4
    assert doc.turns[-2].content == "New question"
    assert doc.turns[-1].content == "New answer"
    doc.save.assert_called_once()


@pytest.mark.asyncio
async def test_update_caps_turns_at_max_turns():
    """After max_turns exchanges, oldest turns are dropped."""
    svc = _service(max_turns=2)
    existing_turns = [
        _turn("user", "Q1"), _turn("assistant", "A1"),
        _turn("user", "Q2"), _turn("assistant", "A2"),
    ]
    doc = _make_doc(existing_turns)
    with _patch_find(doc):
        await svc.update("u1:conv1", "Q3", "A3", owner_id="u1")
    # max_turns=2 → cap at 4 raw turns; adding Q3/A3 takes it to 6, trim to 4
    assert len(doc.turns) == 4
    assert doc.turns[0].content == "Q2"


@pytest.mark.asyncio
async def test_update_skips_empty_query():
    """update() with blank query must not touch the database."""
    svc = _service()
    find_mock = AsyncMock(return_value=None)
    with patch(f"{MODULE}.ChatMemory.find_one", find_mock):
        await svc.update("u1:conv1", "   ", "Some answer", owner_id="u1")
    find_mock.assert_not_called()


# ── clear ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_deletes_existing_document():
    """clear() must call delete() on the found document."""
    svc = _service()
    doc = _make_doc([_turn("user", "Q"), _turn("assistant", "A")])
    with _patch_find(doc):
        await svc.clear("u1:conv1")
    doc.delete.assert_called_once()


@pytest.mark.asyncio
async def test_clear_is_idempotent_when_no_document():
    """clear() on a non-existent session must not raise."""
    svc = _service()
    with _patch_find(None):
        await svc.clear("u1:conv1")  # must not raise


# ── get_history ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_history_returns_all_turns():
    """get_history() must return all stored turns in insertion order."""
    svc = _service()
    turns = [
        _turn("user", "Q1"), _turn("assistant", "A1"),
        _turn("user", "Q2"), _turn("assistant", "A2"),
    ]
    doc = _make_doc(turns)
    with _patch_find(doc):
        history = await svc.get_history("u1:conv1")
    assert len(history) == 4
    assert history[0].content == "Q1"
    assert history[-1].content == "A2"


@pytest.mark.asyncio
async def test_get_history_returns_empty_list_when_no_document():
    """get_history() must return [] for an unknown session."""
    svc = _service()
    with _patch_find(None):
        history = await svc.get_history("u1:nonexistent")
    assert history == []


# ── _session_id ────────────────────────────────────────────────────────────────

def test_session_id_format():
    assert MemoryService._session_id("user42", "conv-abc") == "user42:conv-abc"


# ── build_context (compat wrapper) ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_context_delegates_to_get_context():
    """build_context() should surface per-turn history via new store."""
    from src.rag.types import RetrievalScope
    svc = _service()
    doc = _make_doc([_turn("user", "Hi"), _turn("assistant", "Hello")])
    scope = RetrievalScope(owner_id="u1", collection_id="col1")
    with _patch_find(doc):
        ctx = await svc.build_context(scope=scope, conversation_id="conv1")
    assert "User: Hi" in ctx
    assert "Assistant: Hello" in ctx
