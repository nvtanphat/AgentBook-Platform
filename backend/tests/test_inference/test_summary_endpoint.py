from __future__ import annotations

from fastapi.testclient import TestClient

from src.core.config import get_settings
from src.dependencies import get_summary_service
from src.main import app
from src.schemas.query import SummaryResponse


class FakeSummaryService:
    async def summarize(self, request):
        return SummaryResponse(summary="Grounded summary", citations=[], confidence=0.88)


def test_summary_endpoint_uses_service_override(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_summary_service] = lambda: FakeSummaryService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/query/summarize",
                json={
                    "owner_id": "user_demo",
                    "collection_id": "65f000000000000000000002",
                    "scope": "collection",
                },
            )
        assert response.status_code == 200
        assert response.json()["data"]["summary"] == "Grounded summary"
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
