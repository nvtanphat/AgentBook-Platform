from __future__ import annotations

import logging

from beanie import PydanticObjectId

from src.models.chat_memory import ChatSummaryMemory
from src.models.common import utc_now
from src.models.query_log import QueryLog
from src.rag.types import RetrievalScope

logger = logging.getLogger(__name__)

SHORT_TERM_TURNS = 3
SUMMARY_AFTER_TURNS = 6
MAX_MEMORY_CHARS = 1800
MAX_SUMMARY_CHARS = 900
MAX_MESSAGE_CHARS = 260


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
            sections.append("Summary memory:\n" + summary_doc.summary.strip())
        if recent:
            lines = ["Short-term memory:"]
            for item in reversed(recent):
                lines.append(f"- User: {self._trim(item.query, MAX_MESSAGE_CHARS)}")
                lines.append(f"  Assistant: {self._trim(item.answer, MAX_MESSAGE_CHARS)}")
            sections.append("\n".join(lines))
        return self._trim("\n\n".join(sections), MAX_MEMORY_CHARS)

    async def update_after_query(self, *, scope: RetrievalScope, conversation_id: str) -> None:
        """Refresh extractive summary memory from older turns without adding LLM latency."""
        try:
            collection_oid = self._collection_oid(scope)
            logs = await self._recent_logs(
                owner_id=scope.owner_id,
                collection_id=collection_oid,
                conversation_id=conversation_id,
                limit=SUMMARY_AFTER_TURNS + SHORT_TERM_TURNS,
            )
            if len(logs) <= SUMMARY_AFTER_TURNS:
                return

            older_logs = list(reversed(logs[SHORT_TERM_TURNS:]))
            summary_lines = [
                f"- {self._trim(item.query, 140)} -> {self._trim(item.answer, 220)}"
                for item in older_logs
                if item.query.strip() and item.answer.strip()
            ]
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
