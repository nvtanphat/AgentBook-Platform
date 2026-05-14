from __future__ import annotations

from fastapi.testclient import TestClient

from src.core.config import get_settings
from src.dependencies import get_admin_service
from src.main import app
from src.schemas.admin import AdminMetricsResponse, FeedbackResponse, QueryStats, RetrievalStats


class FakeAdminService:
    async def metrics(self):
        return AdminMetricsResponse(
            total_docs=3,
            failed_jobs=1,
            indexed_docs=2,
            query_stats=QueryStats(total_queries=5, refused_queries=1, average_confidence=0.8, average_latency_ms=120.0),
            retrieval_stats=RetrievalStats(average_top_k=5.0, average_sources_used=2.0, average_retrieval_time_ms=75.0),
            feedback_count=4,
        )

    async def log_feedback(self, request):
        return FeedbackResponse(feedback_id="fb-1", query_log_id=request.query_log_id, rating=request.rating)


def test_admin_metrics_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_admin_service] = lambda: FakeAdminService()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/admin/metrics")
        assert response.status_code == 200
        assert response.json()["data"]["total_docs"] == 3
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_admin_feedback_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_admin_service] = lambda: FakeAdminService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/admin/feedback",
                json={"owner_id": "user_demo", "query_log_id": "65f000000000000000000003", "rating": "helpful"},
            )
        assert response.status_code == 201
        assert response.json()["data"]["rating"] == "helpful"
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_admin_metrics_requires_valid_admin_token_when_auth_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    monkeypatch.setenv("AGENTBOOK_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("AGENTBOOK_API_KEY", "secret")
    get_settings.cache_clear()
    app.dependency_overrides[get_admin_service] = lambda: FakeAdminService()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/admin/metrics")
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
