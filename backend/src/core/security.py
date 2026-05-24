from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


class UploadValidationError(ValueError):
    """Raised when an uploaded file violates upload safety rules."""


ZIP_BASED_EXTENSIONS = {"docx", "pptx", "xlsx"}
OLE_COMPOUND_EXTENSIONS = {"xls"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
TEXT_EXTENSIONS = {"csv"}
UPLOAD_HEAD_BYTES = 4096

MIME_ALLOWLIST: dict[str, set[str]] = {
    "pdf": {"application/pdf"},
    "png": {"image/png"},
    "jpg": {"image/jpeg"},
    "jpeg": {"image/jpeg"},
    "docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"},
    "pptx": {"application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/zip"},
    "xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/zip"},
    "xls": {"application/vnd.ms-excel", "application/octet-stream"},
    "csv": {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"},
    # Audio MIME types (browser MediaRecorder + common encoders)
    "mp3": {"audio/mpeg", "audio/mp3", "application/octet-stream"},
    "wav": {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"},
    "m4a": {"audio/mp4", "audio/x-m4a", "audio/m4a", "application/octet-stream"},
    "ogg": {"audio/ogg", "application/ogg", "application/octet-stream"},
    "flac": {"audio/flac", "audio/x-flac", "application/octet-stream"},
    "webm": {"audio/webm", "video/webm", "application/octet-stream"},
    "aac": {"audio/aac", "audio/x-aac", "application/octet-stream"},
}


@dataclass(frozen=True)
class StreamedUpload:
    temp_path: Path
    size_bytes: int
    checksum_sha256: str
    head: bytes


def normalized_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower().lstrip(".")
    if not extension:
        raise UploadValidationError("Uploaded file must have an extension")
    return extension


def safe_scope_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise UploadValidationError("Scope segment is empty after sanitization")
    return cleaned[:80]


async def read_upload_bytes(file: UploadFile, max_size_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size_bytes:
            raise UploadValidationError("Uploaded file exceeds configured size limit")
        chunks.append(chunk)
    if total == 0:
        raise UploadValidationError("Uploaded file is empty")
    return b"".join(chunks)


async def stream_upload_to_temp(file: UploadFile, max_size_bytes: int, temp_dir: Path) -> StreamedUpload:
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = ensure_child_path(temp_dir, temp_dir / f"{uuid4().hex}.upload")
    digest = hashlib.sha256()
    head = bytearray()
    total = 0
    try:
        with temp_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_size_bytes:
                    raise UploadValidationError("Uploaded file exceeds configured size limit")
                digest.update(chunk)
                if len(head) < UPLOAD_HEAD_BYTES:
                    head.extend(chunk[: UPLOAD_HEAD_BYTES - len(head)])
                await asyncio.to_thread(handle.write, chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    if total == 0:
        temp_path.unlink(missing_ok=True)
        raise UploadValidationError("Uploaded file is empty")
    return StreamedUpload(
        temp_path=temp_path,
        size_bytes=total,
        checksum_sha256=digest.hexdigest(),
        head=bytes(head),
    )


def validate_upload_head(
    *,
    filename: str,
    content_type: str | None,
    head: bytes,
    allowed_extensions: set[str],
) -> str:
    extension = normalized_extension(filename)
    if extension not in allowed_extensions:
        raise UploadValidationError(f"File extension .{extension} is not allowed")

    allowed_mime_types = MIME_ALLOWLIST.get(extension, set())
    if content_type and allowed_mime_types and content_type not in allowed_mime_types:
        raise UploadValidationError(f"MIME type {content_type} is not allowed for .{extension}")

    if extension == "pdf" and not head.startswith(b"%PDF"):
        raise UploadValidationError("PDF magic bytes are invalid")
    if extension == "png" and not head.startswith(b"\x89PNG\r\n\x1a\n"):
        raise UploadValidationError("PNG magic bytes are invalid")
    if extension in {"jpg", "jpeg"} and not head.startswith(b"\xff\xd8\xff"):
        raise UploadValidationError("JPEG magic bytes are invalid")
    if extension in ZIP_BASED_EXTENSIONS and not head.startswith(b"PK"):
        raise UploadValidationError(f"{extension.upper()} file must have ZIP container magic bytes")
    if extension in OLE_COMPOUND_EXTENSIONS and not head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        raise UploadValidationError(f"{extension.upper()} file must have OLE compound document magic bytes")
    if extension in TEXT_EXTENSIONS:
        try:
            head.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadValidationError("CSV must be UTF-8 text for MVP ingestion") from exc

    # Audio magic bytes — best-effort (browsers often send octet-stream)
    if extension == "mp3":
        # MP3 may start with ID3 tag or frame sync (0xFFFB/0xFFE3)
        if not (head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)):
            raise UploadValidationError("MP3 magic bytes are invalid")
    elif extension == "wav":
        if not (head.startswith(b"RIFF") and b"WAVE" in head[:16]):
            raise UploadValidationError("WAV magic bytes are invalid")
    elif extension == "flac":
        if not head.startswith(b"fLaC"):
            raise UploadValidationError("FLAC magic bytes are invalid")
    elif extension == "ogg":
        if not head.startswith(b"OggS"):
            raise UploadValidationError("OGG magic bytes are invalid")
    elif extension in {"m4a", "aac"}:
        # M4A: ftyp box at offset 4; AAC: ADTS sync 0xFFF
        if not (b"ftyp" in head[:16] or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xF0) == 0xF0)):
            raise UploadValidationError(f"{extension.upper()} magic bytes are invalid")
    elif extension == "webm":
        # WebM = EBML container, starts with 0x1A45DFA3
        if not head.startswith(b"\x1a\x45\xdf\xa3"):
            raise UploadValidationError("WebM magic bytes are invalid")

    return extension


def validate_upload_bytes(
    *,
    filename: str,
    content_type: str | None,
    payload: bytes,
    allowed_extensions: set[str],
) -> str:
    return validate_upload_head(
        filename=filename,
        content_type=content_type,
        head=payload[:UPLOAD_HEAD_BYTES],
        allowed_extensions=allowed_extensions,
    )


def ensure_child_path(parent: Path, candidate: Path) -> Path:
    parent_resolved = parent.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise UploadValidationError("Resolved upload path escapes configured storage root") from exc
    return candidate_resolved
