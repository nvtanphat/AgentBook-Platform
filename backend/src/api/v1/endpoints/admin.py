from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.dependencies import get_admin_service
from src.schemas.admin import AdminMetricsResponse, FeedbackRequest, FeedbackResponse
from src.schemas.common import APIResponse
from src.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"])


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
