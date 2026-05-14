from __future__ import annotations

from fastapi.testclient import TestClient

from src.core.config import get_settings
from src.main import app


def test_evaluation_embed_requires_admin_token_when_auth_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    monkeypatch.setenv("AGENTBOOK_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("AGENTBOOK_API_KEY", "secret")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/evaluation/embed", json={"texts": ["hello"]})
        assert response.status_code == 401
    finally:
        get_settings.cache_clear()


def test_evaluation_ragas_rejects_empty_batch(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/evaluation/ragas", json={"samples": []})
        assert response.status_code == 422
    finally:
        get_settings.cache_clear()
