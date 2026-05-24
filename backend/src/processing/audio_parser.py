"""Audio file parser using faster-whisper.

Transcribes audio (mp3, wav, m4a, etc.) to text segments with timestamps,
then groups segments into "pages" of ~60 seconds each so the rest of the
pipeline (chunking, embedding, retrieval) can operate unchanged.

Each block carries `start_seconds` and `end_seconds` in `extra` so citations
can deep-link back to the audio position.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path

from src.core.config import Settings
from src.processing.types import BlockType, ParsedBlock, ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# Light heuristic post-corrections for common Whisper Vietnamese mistakes.
# Pattern-based only (NOT hardcoded entity list) — fixes word-level transcription
# errors that occur across any domain.
_VN_TRANSCRIPTION_FIXES: list[tuple[re.Pattern, str]] = [
    # "giữ liệu" → "dữ liệu" (data) — extremely common Whisper VN error
    (re.compile(r"\bgi[ữu]\s+li[ệe]u\b", re.IGNORECASE), "dữ liệu"),
    # "truyền đổi" → "chuyển đổi" (convert)
    (re.compile(r"\btruy[ềe]n\s+đ[ổo]i\b", re.IGNORECASE), "chuyển đổi"),
    # "phát biểu văn bản" → "phát văn bản" (speech from text) — common mishear
    (re.compile(r"\bph[áa]t\s+bi[ểe]u\s+v[ăa]n\s+b[ảa]n\b", re.IGNORECASE), "phát văn bản"),
    # Double-space cleanup
    (re.compile(r"\s{2,}"), " "),
]


def _apply_vn_corrections(text: str) -> str:
    """Pattern-based heuristic corrections for Whisper Vietnamese errors."""
    if not text:
        return text
    for pattern, replacement in _VN_TRANSCRIPTION_FIXES:
        text = pattern.sub(replacement, text)
    return text.strip()

AUDIO_EXTENSIONS = frozenset({"mp3", "wav", "m4a", "ogg", "flac", "webm", "aac"})


class AudioParser:
    """Lazy-singleton faster-whisper transcriber.

    Model is loaded on first use and reused — avoids repeated 1-2s startup
    cost when indexing multiple audio files.
    """

    _model = None
    _model_name: str | None = None
    _model_lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _get_model(self):
        # Reload when config model name changes (e.g. user upgraded small → medium)
        requested = self.settings.audio_whisper_model
        if AudioParser._model is None or AudioParser._model_name != requested:
            with AudioParser._model_lock:
                if AudioParser._model is None or AudioParser._model_name != requested:
                    try:
                        from faster_whisper import WhisperModel
                    except ImportError as exc:
                        raise RuntimeError(
                            "faster-whisper not installed. Run: pip install faster-whisper"
                        ) from exc
                    logger.info(
                        "Loading faster-whisper model",
                        extra={"model": requested, "device": self.settings.audio_whisper_device},
                    )
                    AudioParser._model = WhisperModel(
                        requested,
                        device=self.settings.audio_whisper_device,
                        compute_type=self.settings.audio_whisper_compute_type,
                    )
                    AudioParser._model_name = requested
        return AudioParser._model

    def parse(self, path: Path, *, language: str = "vi") -> ParsedDocument:
        """Transcribe audio file → ParsedDocument with timestamped blocks.

        Each Whisper segment becomes 1 ParsedBlock. Blocks are grouped into
        "pages" of ~60 seconds for compatibility with the page-based pipeline.
        """
        model = self._get_model()
        ext = path.suffix.lstrip(".").lower()
        logger.info(
            "Transcribing audio",
            extra={"file": path.name, "ext": ext, "size_mb": path.stat().st_size / (1024 * 1024)},
        )

        # Whisper accepts auto-detect when language=None.
        # Map "unknown" / "auto" / empty → None so Whisper detects language itself.
        whisper_lang = None if language in (None, "", "auto", "unknown", "mixed") else language
        segments_iter, info = model.transcribe(
            str(path),
            language=whisper_lang,
            beam_size=self.settings.audio_whisper_beam_size,
            vad_filter=self.settings.audio_whisper_vad_filter,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # Materialize the generator (Whisper streams segments lazily)
        segments = list(segments_iter)
        if not segments:
            logger.warning("Whisper returned 0 segments — audio may be silent or unreadable", extra={"file": path.name})
            return ParsedDocument(
                source_path=str(path),
                file_type=ext,
                language=info.language or language,
                pages=[],
                warnings=["Audio transcription returned no segments. File may be silent or corrupted."],
                extra={"parser": "audio_whisper", "duration": info.duration if info else 0.0},
            )

        detected_lang = info.language or language

        # Group segments into pages of ~60s each
        page_window_seconds = 60.0
        pages: list[ParsedPage] = []
        current_blocks: list[ParsedBlock] = []
        current_page_start = 0.0
        global_index = 0

        def flush_page():
            nonlocal current_blocks
            if current_blocks:
                pages.append(ParsedPage(page_number=len(pages) + 1, blocks=current_blocks))
                current_blocks = []

        is_vietnamese = (detected_lang == "vi")
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            # Apply pattern-based corrections for known Vietnamese Whisper errors
            if is_vietnamese:
                text = _apply_vn_corrections(text)
            # New page when crossing the window boundary
            if seg.start - current_page_start >= page_window_seconds and current_blocks:
                flush_page()
                current_page_start = seg.start
            block = ParsedBlock(
                block_id=f"audio-{global_index}",
                block_index=global_index,
                block_type=BlockType.PARAGRAPH.value,
                content=text,
                page_number=len(pages) + 1,
                reading_order=global_index,
                language=detected_lang,
                source="audio_whisper",
                extra={
                    "start_seconds": float(seg.start),
                    "end_seconds": float(seg.end),
                    "audio_file": path.name,
                    "parse_method": "audio_whisper",
                },
            )
            current_blocks.append(block)
            global_index += 1

        flush_page()

        logger.info(
            "Audio transcription complete",
            extra={
                "file": path.name,
                "duration": info.duration,
                "language": detected_lang,
                "segments": len(segments),
                "blocks": global_index,
                "pages": len(pages),
            },
        )

        return ParsedDocument(
            source_path=str(path),
            file_type=ext,
            language=detected_lang,
            pages=pages,
            warnings=[],
            extra={
                "parser": "audio_whisper",
                "duration": float(info.duration),
                "whisper_model": self.settings.audio_whisper_model,
            },
        )
