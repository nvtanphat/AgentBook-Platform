from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.core.rate_limit import limiter
from src.dependencies import get_query_service, get_study_guide_service, get_summary_service, verify_owner_access
from src.schemas.common import APIResponse
from src.schemas.query import (
    CompareRequest,
    CompareResponse,
    QueryRequest,
    QueryResponse,
    StudyGuideRequest,
    StudyGuideResponse,
    SummaryRequest,
    SummaryResponse,
)
from src.services.query_service import QueryService
from src.services.study_guide_service import StudyGuideService
from src.services.summary_service import SummaryService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/ask", response_model=APIResponse[QueryResponse])
@limiter.limit("15/minute")
async def ask(
    request: Request,
    body: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> APIResponse[QueryResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await query_service.ask(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query pipeline error ({type(exc).__name__}): {exc}",
        ) from exc
    return APIResponse(success=True, message="Query answered successfully", data=result, error=None)


@router.post("/compare", response_model=APIResponse[CompareResponse])
@limiter.limit("10/minute")
async def compare(
    request: Request,
    body: CompareRequest,
    query_service: QueryService = Depends(get_query_service),
) -> APIResponse[CompareResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await query_service.compare(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Compare pipeline error ({type(exc).__name__}): {exc}") from exc
    return APIResponse(success=True, message="Comparison generated successfully", data=result, error=None)


@router.post("/summarize", response_model=APIResponse[SummaryResponse])
async def summarize(
    request: Request,
    body: SummaryRequest,
    summary_service: SummaryService = Depends(get_summary_service),
) -> APIResponse[SummaryResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await summary_service.summarize(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Summary pipeline error ({type(exc).__name__}): {exc}") from exc
    return APIResponse(success=True, message="Summary generated successfully", data=result, error=None)


@router.post("/study-guide", response_model=APIResponse[StudyGuideResponse])
async def study_guide(
    request: Request,
    body: StudyGuideRequest,
    study_guide_service: StudyGuideService = Depends(get_study_guide_service),
) -> APIResponse[StudyGuideResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await study_guide_service.build(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Study guide pipeline error ({type(exc).__name__}): {exc}") from exc
    return APIResponse(success=True, message="Study guide generated successfully", data=result, error=None)
