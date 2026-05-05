from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.runtime_config import get_all_overrides, get_override, set_override
from src.dependencies import get_admin_service, get_settings
from src.schemas.admin import AdminMetricsResponse, FeedbackRequest, FeedbackResponse
from src.schemas.common import APIResponse
from src.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"])


class PipelineSettingsPatch(BaseModel):
    contextual_retrieval_enabled: bool | None = None


@router.get("/settings")
async def get_pipeline_settings(settings=Depends(get_settings)) -> dict:
    return {
        "contextual_retrieval_enabled": bool(
            get_override("contextual_retrieval_enabled", settings.contextual_retrieval_enabled)
        ),
        "contextual_retrieval_concurrency": int(settings.contextual_retrieval_concurrency),
    }


@router.patch("/settings")
async def update_pipeline_settings(body: PipelineSettingsPatch) -> dict:
    if body.contextual_retrieval_enabled is not None:
        set_override("contextual_retrieval_enabled", body.contextual_retrieval_enabled)
    return {"ok": True, "overrides": get_all_overrides()}


@router.get("/metrics", response_model=APIResponse[AdminMetricsResponse])
async def metrics(admin_service: AdminService = Depends(get_admin_service)) -> APIResponse[AdminMetricsResponse]:
    result = await admin_service.metrics()
    return APIResponse(success=True, message="Admin metrics loaded successfully", data=result, error=None)


@router.post("/feedback", response_model=APIResponse[FeedbackResponse], status_code=status.HTTP_201_CREATED)
async def feedback(
    request: FeedbackRequest,
    admin_service: AdminService = Depends(get_admin_service),
) -> APIResponse[FeedbackResponse]:
    try:
        result = await admin_service.log_feedback(request)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return APIResponse(success=True, message="Feedback logged successfully", data=result, error=None)
