"""Semantic cache for query → answer pairs.

When a user asks a question similar to a previously-answered one (cosine ≥ threshold),
return the cached answer instead of running the full RAG pipeline.

Scope: cache keyed by (owner_id, collection_id) to prevent cross-tenant leaks.
Storage: Redis hash with query_embedding + serialized QueryResponse.
Eviction: LRU via Redis EXPIRE (default 1h TTL) + soft cap (200 entries per scope).
"""
from __future__ import annotations

import json
import logging
import math
import struct
from typing import Any

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class SemanticQueryCache:
    """Redis-backed semantic cache for QueryResponse objects.

    Key layout:
      sqc:{owner_id}:{collection_id}:{index}  → JSON {emb: [...], response: {...}, query: str}
      sqc:meta:{owner_id}:{collection_id}     → list of indices for LRU eviction
    """

    def __init__(
        self,
        *,
        redis_url: str,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.93,
        max_entries_per_scope: int = 200,
        key_prefix: str = "sqc",
    ) -> None:
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold
        self.max_entries = max_entries_per_scope
        self.prefix = key_prefix
        self.enabled = False
        try:
            import redis as _redis
            self._redis = _redis.from_url(redis_url, decode_responses=False)
            # Probe connection
            self._redis.ping()
            self.enabled = True
            logger.info(
                "Semantic query cache initialized",
                extra={"ttl": ttl_seconds, "threshold": similarity_threshold, "url": redis_url},
            )
        except Exception as exc:
            logger.warning(
                "Semantic query cache disabled — Redis unavailable",
                extra={"error": str(exc), "url": redis_url},
            )

    # ── Encoding ───────────────────────────────────────────────────────────

    @staticmethod
    def _encode_embedding(emb: list[float]) -> bytes:
        """Pack float list to bytes (4 bytes/float). Smaller than JSON."""
        return struct.pack(f"{len(emb)}f", *emb)

    @staticmethod
    def _decode_embedding(data: bytes) -> list[float]:
        n = len(data) // 4
        return list(struct.unpack(f"{n}f", data))

    def _scope_key(self, owner_id: str, collection_id: str | None, answer_language: str | None = None) -> str:
        # answer_language is part of the scope: BGE-M3 embeds a VI query and its
        # EN translation almost identically, so without this a VI question would
        # hit an EN-cached answer (and vice versa) and be returned in the wrong
        # language. Separate caches per answer language.
        return f"{self.prefix}:{owner_id}:{collection_id or 'none'}:{answer_language or 'auto'}"

    # ── Public API ─────────────────────────────────────────────────────────

    def lookup(
        self,
        *,
        owner_id: str,
        collection_id: str | None,
        query_embedding: list[float],
        answer_language: str | None = None,
    ) -> dict[str, Any] | None:
        """Return cached QueryResponse dict if semantically similar query found."""
        if not self.enabled or not query_embedding:
            return None
        try:
            scope = self._scope_key(owner_id, collection_id, answer_language)
            pattern = f"{scope}:*"
            best_score = 0.0
            best_payload: dict[str, Any] | None = None
            for key in self._redis.scan_iter(match=pattern, count=200):
                raw = self._redis.get(key)
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except Exception:
                    continue
                cached_emb = entry.get("emb")
                if not cached_emb:
                    continue
                score = _cosine(query_embedding, cached_emb)
                if score > best_score:
                    best_score = score
                    best_payload = entry
            if best_payload and best_score >= self.threshold:
                logger.info(
                    "Semantic cache HIT",
                    extra={"score": best_score, "query": best_payload.get("query", "")[:60]},
                )
                return best_payload.get("response")
            return None
        except Exception as exc:
            logger.warning("Semantic cache lookup failed", extra={"error": str(exc)})
            return None

    def store(
        self,
        *,
        owner_id: str,
        collection_id: str | None,
        query: str,
        query_embedding: list[float],
        response: dict[str, Any],
        answer_language: str | None = None,
    ) -> None:
        """Cache a (query, embedding, response) tuple under the given scope."""
        if not self.enabled or not query_embedding:
            return
        try:
            # Don't cache refused answers — they may become valid after re-index
            if response.get("was_refused"):
                return
            scope = self._scope_key(owner_id, collection_id, answer_language)
            # Use timestamp-based suffix; Redis TTL handles eviction
            import time
            idx = int(time.time() * 1000)
            key = f"{scope}:{idx}"
            payload = {
                "query": query[:500],
                "emb": query_embedding,
                "response": response,
            }
            self._redis.setex(key, self.ttl, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logger.warning("Semantic cache store failed", extra={"error": str(exc)})

    def invalidate_scope(self, *, owner_id: str, collection_id: str | None) -> int:
        """Drop all cached entries for a scope (call when documents change)."""
        if not self.enabled:
            return 0
        try:
            # Language-agnostic: scope keys carry a trailing answer-language
            # segment (…:vi, …:en, …:auto), so match every language for this
            # owner/collection when invalidating after a document change.
            scope = f"{self.prefix}:{owner_id}:{collection_id or 'none'}"
            pattern = f"{scope}:*"
            count = 0
            for key in self._redis.scan_iter(match=pattern, count=200):
                self._redis.delete(key)
                count += 1
            if count > 0:
                logger.info("Semantic cache scope invalidated", extra={"scope": scope, "deleted": count})
            return count
        except Exception as exc:
            logger.warning("Semantic cache invalidate failed", extra={"error": str(exc)})
            return 0

    def stats(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        try:
            pattern = f"{self.prefix}:*"
            count = sum(1 for _ in self._redis.scan_iter(match=pattern, count=500))
            return {"enabled": True, "entries": count, "ttl": self.ttl, "threshold": self.threshold}
        except Exception as exc:
            return {"enabled": True, "error": str(exc)}
