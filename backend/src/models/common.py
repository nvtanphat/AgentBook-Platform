from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum


def utc_now() -> datetime:
    return datetime.now(UTC)


class PipelineStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class JobType(StrEnum):
    UPLOAD = "upload"
    PARSE = "parse"
    INDEX = "index"
    PARSE_INDEX = "parse_index"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    MIXED = "mixed"


class SourceLanguage(StrEnum):
    VI = "vi"
    EN = "en"
    UNKNOWN = "unknown"
