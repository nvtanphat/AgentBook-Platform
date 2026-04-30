from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from src.core.config import Settings


def _is_local_path(url: str) -> bool:
    """Return True if url looks like a filesystem path rather than an HTTP URL."""
    if url in (":memory:", ""):
        return False
    p = Path(url)
    # Absolute path or relative path starting with . or ..
    return p.is_absolute() or url.startswith(".") or url.startswith("data/") or url.startswith("data\\")


def build_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    if _is_local_path(settings.qdrant_url):
        path = Path(settings.qdrant_url)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if settings.qdrant_api_key:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=settings.qdrant_timeout_seconds)
    return QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)


_CACHED_CLIENT: QdrantClient | None = None
_CACHED_KEY: tuple[str, str | None, str] | None = None


def get_cached_qdrant_client(qdrant_url: str, qdrant_api_key: str | None, timeout_seconds: int = 60) -> QdrantClient:
    global _CACHED_CLIENT, _CACHED_KEY
    key = (qdrant_url, qdrant_api_key, str(timeout_seconds))
    if _CACHED_CLIENT is not None and _CACHED_KEY == key:
        return _CACHED_CLIENT
    if qdrant_url == ":memory:":
        _CACHED_CLIENT = QdrantClient(location=":memory:")
    elif _is_local_path(qdrant_url):
        path = Path(qdrant_url)
        path.mkdir(parents=True, exist_ok=True)
        _CACHED_CLIENT = QdrantClient(path=str(path))
    elif qdrant_api_key:
        _CACHED_CLIENT = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=timeout_seconds)
    else:
        _CACHED_CLIENT = QdrantClient(url=qdrant_url, timeout=timeout_seconds)
    _CACHED_KEY = key
    return _CACHED_CLIENT


def get_qdrant_client_for_settings(settings: Settings) -> QdrantClient:
    return get_cached_qdrant_client(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_timeout_seconds)


def close_cached_qdrant_client() -> None:
    global _CACHED_CLIENT, _CACHED_KEY
    if _CACHED_CLIENT is not None:
        _CACHED_CLIENT.close()
    _CACHED_CLIENT = None
    _CACHED_KEY = None
