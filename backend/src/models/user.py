from __future__ import annotations

from datetime import datetime

from beanie import Document
from pydantic import EmailStr, Field
from pymongo import IndexModel

from src.models.common import utc_now


class User(Document):
    """Application user account.

    `user_id` is the canonical scope identifier used everywhere as `owner_id`
    (collections, materials, queries). For new accounts we generate a stable
    slug-style id from email. Keeping `user_id` separate from MongoDB `_id`
    means existing data (with owner_id="user_demo" etc.) stays addressable.
    """

    user_id: str = Field(min_length=3, max_length=64)
    email: EmailStr
    display_name: str = Field(default="", max_length=128)
    password_hash: bytes
    role: str = Field(default="user")  # user | admin
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    last_login_at: datetime | None = None

    class Settings:
        name = "users"
        indexes = [
            IndexModel([("user_id", 1)], name="users_user_id_unique", unique=True),
            IndexModel([("email", 1)], name="users_email_unique", unique=True),
        ]
