from __future__ import annotations

import logging

from beanie import PydanticObjectId

from src.models.chat_memory import ChatSummaryMemory
from src.models.common import utc_now
from src.models.query_log import QueryLog
from src.rag.types import RetrievalScope

logger = logging.getLogger(__name__)

SHORT_TERM_TURNS = 4        # verbatim recent turns fed to LLM
SUMMARY_AFTER_TURNS = 6    # compress older turns into summary after this many total turns
MAX_TURNS_SUMMARIZED = 16  # hard cap: never summarize more than this many older turns
MAX_MEMORY_CHARS = 2000
MAX_SUMMARY_CHARS = 1000
MAX_QUERY_KEY_CHARS = 110  # key phrase extracted from query for each summary line
MAX_ANSWER_KEY_CHARS = 180 # first sentence of answer for each summary line


class MemoryService:
    async def build_context(self, *, scope: RetrievalScope, conversation_id: str) -> str:
        """Return bounded conversation memory for the current scoped chat only."""
        try:
            collection_oid = self._collection_oid(scope)
            summary_doc = await self._get_summary(
                owner_id=scope.owner_id,
                collection_id=collection_oid,
                conversation_id=conversation_id,
            )
            recent = await self._recent_logs(
                owner_id=scope.owner_id,
                collection_id=collection_oid,
                conversation_id=conversation_id,
                limit=SHORT_TERM_TURNS,
            )
        except Exception as exc:
            logger.debug("Memory context unavailable", extra={"error": str(exc), "error_type": type(exc).__name__})
            return ""

        sections: list[str] = []
        if summary_doc and summary_doc.summary.strip():
            sections.append("Lịch sử hội thoại (tóm tắt):\n" + summary_doc.summary.strip())
        if recent:
            lines = ["Tin nhắn gần đây:"]
            for item in reversed(recent):
                lines.append(f"- Người dùng: {self._trim(item.query, 220)}")
                lines.append(f"  Trợ lý: {self._trim(item.answer, 300)}")
            sections.append("\n".join(lines))
        return self._trim("\n\n".join(sections), MAX_MEMORY_CHARS)

    async def update_after_query(self, *, scope: RetrievalScope, conversation_id: str) -> None:
        """Refresh extractive summary memory from older turns. Runs in background — no LLM cost."""
        try:
            collection_oid = self._collection_oid(scope)
            total_limit = min(SUMMARY_AFTER_TURNS + SHORT_TERM_TURNS + MAX_TURNS_SUMMARIZED, 64)
            logs = await self._recent_logs(
                owner_id=scope.owner_id,
                collection_id=collection_oid,
                conversation_id=conversation_id,
                limit=total_limit,
            )
            if len(logs) <= SUMMARY_AFTER_TURNS:
                return

            # logs sorted newest-first; skip the SHORT_TERM_TURNS verbatim ones
            older_logs = list(reversed(logs[SHORT_TERM_TURNS:SHORT_TERM_TURNS + MAX_TURNS_SUMMARIZED]))
            summary_lines: list[str] = []
            for idx, item in enumerate(older_logs, start=1):
                if not item.query.strip() or not item.answer.strip():
                    continue
                q = self._trim(item.query, MAX_QUERY_KEY_CHARS)
                a = self._first_sentence(item.answer, MAX_ANSWER_KEY_CHARS)
                summary_lines.append(f"[{idx}] Q: {q}\n    A: {a}")

            summary = self._trim("\n".join(summary_lines), MAX_SUMMARY_CHARS)
            if not summary:
                return

            summary_doc = await self._get_summary(
                owner_id=scope.owner_id,
                collection_id=collection_oid,
                conversation_id=conversation_id,
            )
            if summary_doc is None:
                summary_doc = ChatSummaryMemory(
                    owner_id=scope.owner_id,
                    collection_id=collection_oid,
                    conversation_id=conversation_id,
                    summary=summary,
                    source_query_count=len(logs),
                )
                await summary_doc.insert()
            else:
                summary_doc.summary = summary
                summary_doc.source_query_count = len(logs)
                summary_doc.updated_at = utc_now()
                await summary_doc.save()
        except Exception as exc:
            logger.debug("Memory summary update skipped", extra={"error": str(exc), "error_type": type(exc).__name__})

    @staticmethod
    def _collection_oid(scope: RetrievalScope) -> PydanticObjectId | None:
        if not scope.collection_id:
            return None
        return PydanticObjectId(scope.collection_id)

    @staticmethod
    async def _get_summary(
        *,
        owner_id: str,
        collection_id: PydanticObjectId | None,
        conversation_id: str,
    ) -> ChatSummaryMemory | None:
        return await ChatSummaryMemory.find_one(
            ChatSummaryMemory.owner_id == owner_id,
            ChatSummaryMemory.collection_id == collection_id,
            ChatSummaryMemory.conversation_id == conversation_id,
        )

    @staticmethod
    async def _recent_logs(
        *,
        owner_id: str,
        collection_id: PydanticObjectId | None,
        conversation_id: str,
        limit: int,
    ) -> list[QueryLog]:
        return await QueryLog.find(
            QueryLog.owner_id == owner_id,
            QueryLog.collection_id == collection_id,
            QueryLog.conversation_id == conversation_id,
        ).sort("-created_at").limit(limit).to_list()

    @staticmethod
    def _trim(value: str, limit: int) -> str:
        text = " ".join(value.split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    @staticmethod
    def _first_sentence(value: str, limit: int) -> str:
        """Extract first meaningful sentence, then trim to limit."""
        text = " ".join(value.split())
        for sep in (".", "!", "?", "\n"):
            pos = text.find(sep)
            if 20 < pos < limit:
                text = text[: pos + 1]
                break
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"
