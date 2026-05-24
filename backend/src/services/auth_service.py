"""Authentication service: register, login, token issuance.

Uses HS256 JWT (compatible with existing _verify_hs256_jwt in dependencies.py).
Passwords hashed with bcrypt (12 rounds default).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import time
from datetime import datetime

import bcrypt
from beanie.operators import Or

from src.core.config import Settings
from src.models.common import utc_now
from src.models.user import User

logger = logging.getLogger(__name__)

# Token lifetimes
ACCESS_TOKEN_TTL_SECONDS = 60 * 60 * 12      # 12h
REFRESH_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


class AuthError(Exception):
    """Raised when authentication fails. Maps to HTTP 401 in endpoint."""


def _slugify_user_id(email: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", email.lower()).strip("_")
    return base[:48] or "user"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign_jwt(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h}.{p}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    s = _b64url_encode(sig)
    return f"{h}.{p}.{s}"


def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12))


def verify_password(plain: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed)
    except Exception:
        return False


class AuthService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def _secret(self) -> str:
        # Reuse the API key as JWT secret. Required when auth is enabled.
        secret = self.settings.api_key or "dev-insecure-secret-change-me"
        return secret

    async def register(
        self, *, email: str, password: str, display_name: str = ""
    ) -> User:
        if len(password) < 6:
            raise AuthError("Mật khẩu phải có ít nhất 6 ký tự.")
        email_norm = email.lower().strip()
        existing = await User.find_one({"email": email_norm})
        if existing:
            raise AuthError("Email đã được sử dụng.")
        user_id = _slugify_user_id(email_norm)
        # Ensure user_id uniqueness
        if await User.find_one({"user_id": user_id}):
            user_id = f"{user_id}_{int(time.time()) % 10000}"
        user = User(
            user_id=user_id,
            email=email_norm,
            display_name=(display_name or email_norm.split("@", 1)[0])[:128],
            password_hash=hash_password(password),
        )
        await user.insert()
        logger.info("User registered", extra={"user_id": user_id, "email": email_norm})
        return user

    async def authenticate(self, *, email_or_id: str, password: str) -> User:
        ident = email_or_id.lower().strip()
        user = await User.find_one(Or(User.email == ident, User.user_id == ident))
        if user is None:
            raise AuthError("Email hoặc mật khẩu không đúng.")
        if not user.is_active:
            raise AuthError("Tài khoản đã bị khóa.")
        if not verify_password(password, user.password_hash):
            raise AuthError("Email hoặc mật khẩu không đúng.")
        # Best-effort last_login update
        try:
            user.last_login_at = utc_now()
            await user.save()
        except Exception:
            pass
        return user

    def issue_access_token(self, user: User) -> str:
        now = int(time.time())
        payload = {
            "sub": user.user_id,
            "owner_id": user.user_id,
            "email": user.email,
            "role": user.role,
            "iat": now,
            "exp": now + ACCESS_TOKEN_TTL_SECONDS,
            "type": "access",
        }
        return _sign_jwt(payload, self._secret)

    def issue_refresh_token(self, user: User) -> str:
        now = int(time.time())
        payload = {
            "sub": user.user_id,
            "iat": now,
            "exp": now + REFRESH_TOKEN_TTL_SECONDS,
            "type": "refresh",
        }
        return _sign_jwt(payload, self._secret)

    async def refresh(self, refresh_token: str) -> tuple[User, str]:
        from src.dependencies import _verify_hs256_jwt
        try:
            payload = _verify_hs256_jwt(token=refresh_token, secret=self._secret)
        except Exception as exc:
            raise AuthError("Refresh token không hợp lệ.") from exc
        if payload.get("type") != "refresh":
            raise AuthError("Token không phải refresh token.")
        exp = payload.get("exp")
        if not exp or float(exp) < time.time():
            raise AuthError("Refresh token đã hết hạn.")
        user_id = payload.get("sub")
        if not user_id:
            raise AuthError("Refresh token thiếu subject.")
        user = await User.find_one({"user_id": user_id})
        if user is None or not user.is_active:
            raise AuthError("Tài khoản không tồn tại hoặc bị khóa.")
        return user, self.issue_access_token(user)


def public_user_payload(user: User) -> dict:
    return {
        "user_id": user.user_id,
        "email": user.email,
        "display_name": user.display_name or user.user_id,
        "role": user.role,
        "created_at": user.created_at.isoformat() if isinstance(user.created_at, datetime) else None,
    }
