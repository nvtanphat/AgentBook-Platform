from __future__ import annotations

import logging

from beanie import PydanticObjectId

from src.core.config import get_settings
from src.models.chat_memory import ChatMemory, ChatTurn
from src.models.chat_memory import ChatSummaryMemory
from src.models.common import utc_now
from src.models.query_log import QueryLog
from src.rag.types import RetrievalScope

logger = logging.getLogger(__name__)

MAX_MEMORY_CHARS = 2000
MAX_QUERY_KEY_CHARS = 220
MAX_ANSWER_KEY_CHARS = 300


class MemoryService:
    def __init__(self, max_turns: int | None = None) -> None:
        if max_turns is not None:
            self._max_turns = max_turns
        else:
            try:
                self._max_turns = get_settings().memory_max_turns
            except Exception:
                self._max_turns = 10

    # ── Primary interface ──────────────────────────────────────────────────────

    async def get_context(self, session_id: str, last_n: int | None = None) -> str:
        """Return the last N turns as 'User: ...\nAssistant: ...' blocks."""
        n = last_n if last_n is not None else self._max_turns
        try:
            doc = await ChatMemory.find_one({"session_id": session_id})
        except Exception as exc:
            logger.debug("Memory get_context failed", extra={"session_id": session_id, "error": str(exc)})
            return ""
        if doc is None or not doc.turns:
            return ""
        turns = doc.turns[-n * 2:]  # each exchange = 2 turns
        lines: list[str] = []
        for turn in turns:
            if turn.role == "user":
                lines.append(f"User: {self._trim(turn.content, MAX_QUERY_KEY_CHARS)}")
            else:
                lines.append(f"Assistant: {self._trim(turn.content, MAX_ANSWER_KEY_CHARS)}")
        return self._trim("\n".join(lines), MAX_MEMORY_CHARS)

    async def update(
        self,
        session_id: str,
        query: str,
        answer: str,
        collection_id: str = "",
        *,
        owner_id: str = "",
    ) -> None:
        """Append a user/assistant exchange and cap to max_turns."""
        if not query.strip():
            return
        try:
            doc = await ChatMemory.find_one({"session_id": session_id})
            new_turns = [
                ChatTurn(role="user", content=query.strip()),
                ChatTurn(role="assistant", content=answer.strip()),
            ]
            if doc is None:
                doc = ChatMemory(
                    owner_id=owner_id,
                    session_id=session_id,
                    collection_id=collection_id,
                    turns=new_turns,
                )
                await doc.insert()
            else:
                doc.turns.extend(new_turns)
                # Cap: keep last max_turns exchanges (2 turns each)
                cap = self._max_turns * 2
                if len(doc.turns) > cap:
                    doc.turns = doc.turns[-cap:]
                doc.updated_at = utc_now()
                await doc.save()
        except Exception as exc:
            logger.debug(
                "Memory update failed",
                extra={"session_id": session_id, "error": str(exc)},
            )

    async def clear(self, session_id: str) -> None:
        """Delete all turns for the given session."""
        try:
            doc = await ChatMemory.find_one({"session_id": session_id})
            if doc is not None:
                await doc.delete()
        except Exception as exc:
            logger.debug(
                "Memory clear failed",
                extra={"session_id": session_id, "error": str(exc)},
            )

    async def get_history(self, session_id: str) -> list[ChatTurn]:
        """Return all stored turns for the session (oldest first)."""
        try:
            doc = await ChatMemory.find_one({"session_id": session_id})
        except Exception as exc:
            logger.debug(
                "Memory get_history failed",
                extra={"session_id": session_id, "error": str(exc)},
            )
            return []
        if doc is None:
            return []
        return list(doc.turns)

    # ── Backward-compat wrappers (used by QueryService) ────────────────────────

    async def build_context(self, *, scope: RetrievalScope, conversation_id: str) -> str:
        """Legacy entry point — delegates to get_context."""
        session_id = self._session_id(scope.owner_id, conversation_id)
        try:
            context = await self.get_context(session_id)
        except Exception as exc:
            logger.debug(
                "build_context fallback to QueryLog",
                extra={"error": str(exc)},
            )
            context = await self._query_log_context(scope=scope, conversation_id=conversation_id)
        if context:
            return context
        # Fallback: pull from QueryLog so existing history isn't lost on first deploy
        return await self._query_log_context(scope=scope, conversation_id=conversation_id)

    async def update_after_query(self, *, scope: RetrievalScope, conversation_id: str) -> None:
        """Legacy entry point called by QueryService without query/answer text.

        This path cannot supply query/answer text so it does nothing. The
        QueryService should call `update()` directly after receiving the
        response. Kept for backward compatibility only.
        """

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _session_id(owner_id: str, conversation_id: str) -> str:
        return f"{owner_id}:{conversation_id}"

    @staticmethod
    def _collection_oid(scope: RetrievalScope) -> PydanticObjectId | None:
        if not scope.collection_id:
            return None
        return PydanticObjectId(scope.collection_id)

    @staticmethod
    async def _query_log_context(*, scope: RetrievalScope, conversation_id: str) -> str:
        """Fallback: build context from QueryLog for sessions that pre-date ChatMemory."""
        try:
            collection_oid: PydanticObjectId | None = None
            if scope.collection_id:
                collection_oid = PydanticObjectId(scope.collection_id)
            recent = await QueryLog.find(
                QueryLog.owner_id == scope.owner_id,
                QueryLog.collection_id == collection_oid,
                QueryLog.conversation_id == conversation_id,
            ).sort("-created_at").limit(3).to_list()
            if not recent:
                return ""
            lines: list[str] = []
            for item in reversed(recent):
                lines.append(f"User: {MemoryService._trim(item.query, MAX_QUERY_KEY_CHARS)}")
                lines.append(f"Assistant: {MemoryService._trim(item.answer, MAX_ANSWER_KEY_CHARS)}")
            return MemoryService._trim("\n".join(lines), MAX_MEMORY_CHARS)
        except Exception:
            return ""

    @staticmethod
    def _trim(value: str, limit: int) -> str:
        text = " ".join(value.split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"
