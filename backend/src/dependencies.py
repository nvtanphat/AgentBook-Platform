from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from functools import lru_cache
from threading import RLock

from fastapi import Depends, HTTPException, Request, status
from qdrant_client import QdrantClient

from src.core.config import Settings, get_settings
from src.rag.vector_store import get_qdrant_client_for_settings
from src.services.admin_service import AdminService
from src.services.material_service import MaterialService
from src.services.query_service import QueryService
from src.services.study_guide_service import StudyGuideService
from src.services.summary_service import SummaryService

_QUERY_SERVICE_LOCK = RLock()
_QUERY_SERVICE: QueryService | None = None


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return get_qdrant_client_for_settings(get_settings())


def get_app_settings() -> Settings:
    return get_settings()


def verify_owner_access(request: Request, owner_id: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not settings.api_auth_enabled:
        return
    if not settings.api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API auth is enabled but AGENTBOOK_API_KEY is not configured")
    authenticated_owner = _owner_from_authorization(request.headers.get("authorization"), secret=settings.api_key)
    if authenticated_owner is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API credentials")
    if authenticated_owner != owner_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="owner_id does not match authenticated owner")


def _owner_from_authorization(header: str | None, *, secret: str) -> str | None:
    if not header or not header.startswith("Bearer "):
        return None
    token = header.removeprefix("Bearer ").strip()
    if token == secret:
        return None
    try:
        payload = _verify_hs256_jwt(token=token, secret=secret)
    except ValueError:
        return None
    owner_id = payload.get("sub") or payload.get("owner_id")
    if not isinstance(owner_id, str) or not owner_id:
        return None
    exp = payload.get("exp")
    if exp is not None:
        try:
            if float(exp) < time.time():
                return None
        except (TypeError, ValueError):
            return None
    return owner_id


def _verify_hs256_jwt(*, token: str, secret: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("token must have three JWT segments")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual = _b64url_decode(parts[2])
    if not hmac.compare_digest(expected, actual):
        raise ValueError("invalid JWT signature")
    header = json.loads(_b64url_decode(parts[0]))
    if header.get("alg") != "HS256":
        raise ValueError("unsupported JWT alg")
    payload = json.loads(_b64url_decode(parts[1]))
    if not isinstance(payload, dict):
        raise ValueError("JWT payload must be an object")
    return payload


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def get_material_service(settings: Settings = Depends(get_app_settings)) -> MaterialService:
    return MaterialService(settings=settings)


def get_query_service() -> QueryService:
    global _QUERY_SERVICE
    with _QUERY_SERVICE_LOCK:
        if _QUERY_SERVICE is None:
            _QUERY_SERVICE = QueryService(settings=get_settings())
        return _QUERY_SERVICE


async def close_query_service() -> None:
    global _QUERY_SERVICE
    with _QUERY_SERVICE_LOCK:
        service = _QUERY_SERVICE
        _QUERY_SERVICE = None
    if service is None:
        return
    close = getattr(service.inference_engine.llm, "close", None)
    if close is not None:
        await close()


def get_summary_service(settings: Settings = Depends(get_app_settings)) -> SummaryService:
    return SummaryService(settings=settings)


def get_study_guide_service(settings: Settings = Depends(get_app_settings)) -> StudyGuideService:
    return StudyGuideService(settings=settings)


def get_admin_service() -> AdminService:
    return AdminService()
