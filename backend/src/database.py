from __future__ import annotations

import logging
from dataclasses import dataclass

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import Settings
from src.models.chunk import Chunk
from src.models.chat_memory import ChatMemory, ChatSummaryMemory
from src.models.collection import KnowledgeCollection
from src.models.feedback import Feedback
from src.models.knowledge_graph import Entity, Event, Relation
from src.models.material import Material, MaterialPageDocument
from src.models.pipeline_job import PipelineJob
from src.models.query_log import QueryLog
from src.models.translation_cache import TranslationCache
from src.models.user import User

logger = logging.getLogger(__name__)


DOCUMENT_MODELS = [
    User,
    KnowledgeCollection,
    Material,
    MaterialPageDocument,
    Chunk,
    PipelineJob,
    TranslationCache,
    Entity,
    Event,
    Relation,
    QueryLog,
    ChatMemory,
    ChatSummaryMemory,
    Feedback,
]


@dataclass
class DatabaseState:
    client: AsyncIOMotorClient | None = None


db_state = DatabaseState()


async def init_database(settings: Settings) -> None:
    if settings.testing:
        logger.info("Skipping MongoDB initialization in testing mode")
        return
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is required to initialize MongoDB Atlas")

    client = AsyncIOMotorClient(settings.mongodb_uri)
    await init_beanie(database=client[settings.mongodb_database], document_models=DOCUMENT_MODELS)
    db_state.client = client
    logger.info("MongoDB/Beanie initialized", extra={"database": settings.mongodb_database})


async def close_database() -> None:
    if db_state.client is not None:
        close = getattr(db_state.client, "close", None)
        if callable(close):
            close()
        db_state.client = None
