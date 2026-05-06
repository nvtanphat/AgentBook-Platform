from __future__ import annotations

import base64
import hashlib
import hmac
import json
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.core.config import get_settings
from src.core.config import Settings
from src.dependencies import get_query_service, verify_owner_access
from src.main import app
from src.schemas.query import AgentTrace, AgentTraceStep, QueryResponse


def make_token(payload: dict, *, secret: str = "secret") -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def enc(value: dict | bytes) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode() if isinstance(value, dict) else value
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    signing_input = f"{enc(header)}.{enc(payload)}"
    signature = hmac.new(secret.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{enc(signature)}"


class FakeQueryService:
    async def ask(self, request):
        assert request.answer_language == "vi"
        return QueryResponse(
            answer="grounded",
            answer_language=request.answer_language,
            query_language="vi",
            source_languages=[],
            citations=[],
            confidence=0.8,
            was_refused=False,
        )

    async def compare(self, request):
        raise AssertionError("not used")

    async def ask_stream(self, request):
        step = AgentTraceStep(name="plan_query", status="completed", query=request.query)
        yield f"event: agent_step\ndata: {step.model_dump_json()}\n\n"
        response = QueryResponse(
            answer="grounded stream",
            answer_language=request.answer_language or "vi",
            query_language="vi",
            source_languages=[],
            citations=[],
            confidence=0.8,
            was_refused=False,
            agent_trace=AgentTrace(plan_type="factual", steps=[step]),
        )
        yield f"event: done\ndata: {response.model_dump_json()}\n\n"


def test_query_ask_endpoint_uses_service_override(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_query_service] = lambda: FakeQueryService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/query/ask",
                json={
                    "owner_id": "user_demo",
                    "collection_id": "65f000000000000000000002",
                    "query": "Dropout là gì?",
                    "answer_language": "vi",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["answer"] == "grounded"
        assert body["data"]["answer_language"] == "vi"
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_query_ask_stream_endpoint_forwards_agentic_events(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_query_service] = lambda: FakeQueryService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/query/ask-stream",
                json={
                    "owner_id": "user_demo",
                    "collection_id": "65f000000000000000000002",
                    "query": "Dropout la gi?",
                    "answer_language": "vi",
                },
            )
        assert response.status_code == 200
        body = response.text
        assert "event: agent_step" in body
        assert '"name":"plan_query"' in body
        assert "event: done" in body
        assert '"agent_trace"' in body
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_owner_access_fails_closed_when_auth_enabled_without_api_key() -> None:
    request = SimpleNamespace(headers={})
    settings = Settings(testing=True, api_auth_enabled=True, api_key=None)

    try:
        verify_owner_access(request, "user_demo", settings)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 500
    else:
        raise AssertionError("auth enabled without API key should fail closed")


def test_owner_access_rejects_spoofed_owner_token() -> None:
    request = SimpleNamespace(headers={"authorization": f"Bearer {make_token({'sub': 'attacker'})}"})
    settings = Settings(testing=True, api_auth_enabled=True, api_key="secret")

    try:
        verify_owner_access(request, "user_demo", settings)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("spoofed owner header should be rejected")


def test_owner_access_accepts_signed_owner_token() -> None:
    request = SimpleNamespace(headers={"authorization": f"Bearer {make_token({'sub': 'user_demo'})}"})
    settings = Settings(testing=True, api_auth_enabled=True, api_key="secret")

    verify_owner_access(request, "user_demo", settings)
