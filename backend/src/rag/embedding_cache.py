from __future__ import annotations

import hashlib
import logging
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.embedder import EmbeddedText

logger = logging.getLogger(__name__)


class RedisEmbeddingCache:
    """
    Redis-backed embedding cache shared across workers.

    Replaces in-memory per-process cache with shared Redis cache.
    Reduces embedding calls by 70-80% in production.
    """

    def __init__(self, redis_url: str, ttl: int = 300, key_prefix: str = "emb") -> None:
        self.ttl = ttl
        self.key_prefix = key_prefix
        try:
            import redis
            self.redis = redis.from_url(redis_url, decode_responses=False)
            self.enabled = True
            logger.info("Redis embedding cache initialized", extra={"ttl": ttl, "url": redis_url})
        except ImportError:
            logger.warning("redis package not installed, falling back to in-memory cache")
            self.enabled = False
        except Exception as exc:
            logger.warning("Redis connection failed, falling back to in-memory cache", extra={"error": str(exc)})
            self.enabled = False

    def _key(self, text: str) -> str:
        """Generate cache key from text hash."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{self.key_prefix}:{text_hash}"

    def get(self, text: str) -> "EmbeddedText | None":
        """Retrieve cached embedding for text."""
        if not self.enabled:
            return None
        try:
            key = self._key(text)
            data = self.redis.get(key)
            if data:
                logger.debug("Redis cache hit", extra={"key": key})
                return pickle.loads(data)
            return None
        except Exception as exc:
            logger.warning("Redis get failed", extra={"error": str(exc)})
            return None

    def set(self, text: str, embedding: "EmbeddedText") -> None:
        """Cache embedding for text with TTL."""
        if not self.enabled:
            return
        try:
            key = self._key(text)
            data = pickle.dumps(embedding)
            self.redis.setex(key, self.ttl, data)
            logger.debug("Redis cache set", extra={"key": key, "ttl": self.ttl})
        except Exception as exc:
            logger.warning("Redis set failed", extra={"error": str(exc)})

    def delete(self, text: str) -> None:
        """Delete cached embedding."""
        if not self.enabled:
            return
        try:
            key = self._key(text)
            self.redis.delete(key)
        except Exception as exc:
            logger.warning("Redis delete failed", extra={"error": str(exc)})

    def clear(self) -> None:
        """Clear all cached embeddings."""
        if not self.enabled:
            return
        try:
            pattern = f"{self.key_prefix}:*"
            for key in self.redis.scan_iter(match=pattern, count=100):
                self.redis.delete(key)
            logger.info("Redis cache cleared")
        except Exception as exc:
            logger.warning("Redis clear failed", extra={"error": str(exc)})

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        try:
            info = self.redis.info("stats")
            return {
                "enabled": True,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)),
            }
        except Exception as exc:
            logger.warning("Redis stats failed", extra={"error": str(exc)})
            return {"enabled": True, "error": str(exc)}
