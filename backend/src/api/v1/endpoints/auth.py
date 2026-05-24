from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from src.core.config import Settings
from src.dependencies import _verify_hs256_jwt, get_app_settings
from src.schemas.common import APIResponse
from src.services.auth_service import AuthError, AuthService, public_user_payload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=128)


class LoginRequest(BaseModel):
    # Accept either email or user_id for login
    email_or_id: str = Field(min_length=3, max_length=128)
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=4096)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"
    user: dict


def get_auth_service(settings: Settings = Depends(get_app_settings)) -> AuthService:
    return AuthService(settings)


async def _resolve_current_user(request: Request, settings: Settings):
    """Pull the bearer token, verify it, return the User record."""
    from src.models.user import User
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    secret = settings.api_key or "dev-insecure-secret-change-me"
    try:
        payload = _verify_hs256_jwt(token=header.removeprefix("Bearer ").strip(), secret=secret)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    if payload.get("type") not in (None, "access"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expected access token")
    user_id = payload.get("sub") or payload.get("owner_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject")
    user = await User.find_one({"user_id": user_id})
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return user


@router.post("/register", response_model=APIResponse[TokenResponse])
async def register(
    body: RegisterRequest,
    auth: AuthService = Depends(get_auth_service),
):
    try:
        user = await auth.register(email=body.email, password=body.password, display_name=body.display_name)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return APIResponse(
        success=True,
        message="Đăng ký thành công",
        data=TokenResponse(
            access_token=auth.issue_access_token(user),
            refresh_token=auth.issue_refresh_token(user),
            user=public_user_payload(user),
        ),
        error=None,
    )


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(
    body: LoginRequest,
    auth: AuthService = Depends(get_auth_service),
):
    try:
        user = await auth.authenticate(email_or_id=body.email_or_id, password=body.password)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return APIResponse(
        success=True,
        message="Đăng nhập thành công",
        data=TokenResponse(
            access_token=auth.issue_access_token(user),
            refresh_token=auth.issue_refresh_token(user),
            user=public_user_payload(user),
        ),
        error=None,
    )


@router.post("/refresh", response_model=APIResponse[TokenResponse])
async def refresh(
    body: RefreshRequest,
    auth: AuthService = Depends(get_auth_service),
):
    try:
        user, access_token = await auth.refresh(body.refresh_token)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return APIResponse(
        success=True,
        message="Token refreshed",
        data=TokenResponse(
            access_token=access_token,
            refresh_token=None,  # client keeps existing refresh until expiry
            user=public_user_payload(user),
        ),
        error=None,
    )


@router.get("/me", response_model=APIResponse[dict])
async def me(
    request: Request,
    settings: Settings = Depends(get_app_settings),
):
    user = await _resolve_current_user(request, settings)
    return APIResponse(
        success=True,
        message="Current user",
        data=public_user_payload(user),
        error=None,
    )
