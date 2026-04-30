from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import unicodedata
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from src.processing.types import BBox, BlockType, DependencyUnavailableError, ParsedBlock, ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)


_VIETNAMESE_SPECIALS = {chr(codepoint) for codepoint in (0x0111, 0x0110, 0x01A1, 0x01A0, 0x01B0, 0x01AF)}
_SUSPECT_LATIN_DIACRITICS = {chr(codepoint) for codepoint in (0x0113, 0x016B, 0x01CE, 0x00E5, 0x012B, 0x014D)}

# Latin Extended lookalikes that PaddleOCR commonly confuses with Vietnamese chars.
# Maps wrong → correct Unicode codepoint.
_VI_LOOKALIKE_MAP = str.maketrans({
    'ē': 'ê',  # ē → ê
    'Ē': 'Ê',  # Ē → Ê
    'ū': 'ư',  # ū → ư
    'Ū': 'Ư',  # Ū → Ư
    'ǎ': 'ă',  # ǎ → ă
    'Ǎ': 'Ă',  # Ǎ → Ă
    'ō': 'ô',  # ō → ô
    'Ō': 'Ô',  # Ō → Ô
    'ī': 'î',  # ī → î
    'Ī': 'Î',  # Ī → Î
    'ǐ': 'í',  # ǐ → í  (common in vi OCR)
    'Ǐ': 'Í',  # Ǐ → Í
    'ǒ': 'ò',  # ǒ → ò
    'Ǒ': 'Ò',  # Ǒ → Ò
    'ĕ': 'è',  # ĕ → è
    'Ĕ': 'È',  # Ĕ → È
    'å': 'ã',  # å → ã (common confusion in scan)
    'Å': 'Ã',  # Å → Ã
})

# Common whole-token OCR corrections for Vietnamese (post-character-fix pass).
# Only corrects unambiguous high-frequency errors seen in scan output.
_VI_TOKEN_FIXES: dict[str, str] = {
    "hp": "hợp",
    "dùng": "dùng",   # keep (already correct)
    "vǎn": "văn",
    "bǎng": "băng",
}


def _has_vietnamese_mark(char: str) -> bool:
    return char in _VIETNAMESE_SPECIALS or any(
        unicodedata.category(part) == "Mn" for part in unicodedata.normalize("NFD", char)
    )


def _workspace_cache_dir(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "cache" / name


def _configure_ocr_cache() -> None:
    cache_dirs = {
        "PADDLE_PDX_CACHE_HOME": _workspace_cache_dir("paddlex"),
        "HF_HOME": _workspace_cache_dir("huggingface"),
        "MODELSCOPE_CACHE": _workspace_cache_dir("modelscope"),
    }
    for name, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "False")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")


# ── Image preprocessing ───────────────────────────────────────────────────────

class _ImagePreprocessor:
    """Deterministic preprocessing pipeline to improve OCR input quality."""

    # Minimum long-edge pixel length before upscaling.
    _MIN_LONG_EDGE = 1600

    @classmethod
    def build_variants(cls, image_path: Path, *, cache_dir: Path) -> dict[str, Path]:
        """Return a dict of {variant_name: preprocessed_path} ready for OCR.

        Always includes 'enhanced' (main improved variant).
        Adds 'binarized' only when the image appears low-contrast.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return {}

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {}

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        variants: dict[str, Path] = {}

        # --- Enhanced variant (always) ---
        enhanced = cls._enhance(gray, h=h, w=w, np=np, cv2=cv2)
        enhanced_path = cls._save(enhanced, image_path, "enhanced", cache_dir)
        if enhanced_path:
            variants["enhanced"] = enhanced_path

        # --- Binarized variant (when image contrast is low) ---
        contrast = float(np.std(gray))
        if contrast < 60:
            binarized = cls._binarize(enhanced, cv2=cv2)
            binarized_path = cls._save(binarized, image_path, "binarized", cache_dir)
            if binarized_path:
                variants["binarized"] = binarized_path

        return variants

    @classmethod
    def _enhance(cls, gray, *, h: int, w: int, np, cv2) -> "np.ndarray":
        """Upscale → denoise → CLAHE → unsharp-mask."""
        # 1. Upscale to at least MIN_LONG_EDGE on the long side for better char detail
        long_edge = max(h, w)
        if long_edge < cls._MIN_LONG_EDGE:
            scale = cls._MIN_LONG_EDGE / long_edge
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # 2. Denoise — light enough not to smear diacritics
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)

        # 3. CLAHE — equalise local contrast so faint text becomes readable
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 4. Unsharp mask — sharpen without amplifying noise
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        gray = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        return gray

    @staticmethod
    def _binarize(gray, *, cv2) -> "cv2.Mat":
        """Adaptive threshold — works well for high-contrast printed documents."""
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25,
            C=10,
        )

    @staticmethod
    def _save(gray_image, source_path: Path, variant_name: str, cache_dir: Path) -> Path | None:
        try:
            import cv2
            cache_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(str(source_path.resolve()).encode()).hexdigest()[:12]
            out_path = cache_dir / f"{source_path.stem}-{digest}-{variant_name}.png"
            if not out_path.exists():
                bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            return out_path
        except Exception:
            logger.exception("Failed to save preprocessed variant '%s'", variant_name)
            return None


# ── Text post-processing ──────────────────────────────────────────────────────

def _normalize_vi_text(text: str) -> str:
    """Fix Latin Extended lookalikes that PaddleOCR substitutes for Vietnamese chars."""
    return text.translate(_VI_LOOKALIKE_MAP)


def _sort_blocks_by_reading_order(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Sort blocks top-to-bottom, left-to-right using bbox centroid.

    Blocks without bbox are placed at the end in original order.
    """
    with_bbox = [(b, (b.bbox.y1 + b.bbox.y2) / 2, (b.bbox.x1 + b.bbox.x2) / 2)
                 for b in blocks if b.bbox is not None]
    without_bbox = [b for b in blocks if b.bbox is None]

    if not with_bbox:
        return blocks

    # Bucket into approximate text lines (within 15px vertical distance → same line)
    LINE_TOLERANCE = 15
    with_bbox.sort(key=lambda t: t[1])  # sort by cy first
    lines: list[list[tuple]] = []
    for item in with_bbox:
        placed = False
        for line in lines:
            if abs(item[1] - line[0][1]) <= LINE_TOLERANCE:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    sorted_blocks: list[ParsedBlock] = []
    for line in lines:
        line.sort(key=lambda t: t[2])  # sort by cx within each line
        sorted_blocks.extend(t[0] for t in line)

    return sorted_blocks + without_bbox


def _deduplicate_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Remove blocks whose content is nearly identical to a nearby block.

    Duplicates arise when multi-variant OCR assigns different variant blocks
    to the same spatial position. We keep the higher-confidence version.
    """
    if len(blocks) <= 1:
        return blocks

    import re as _re
    import unicodedata as _ud

    def _tokens(text: str) -> set[str]:
        """ASCII-normalized token set — diacritics stripped for robust comparison."""
        normalized = _ud.normalize("NFD", text.lower())
        ascii_form = "".join(c for c in normalized if _ud.category(c) != "Mn")
        return set(_re.findall(r"\w+", ascii_form))

    def _sim(a: str, b: str) -> float:
        """Token-level Jaccard on diacritics-stripped text."""
        sa, sb = _tokens(a), _tokens(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    kept: list[ParsedBlock] = [blocks[0]]
    for block in blocks[1:]:
        blk_text = block.content.strip()
        blk_tokens = _tokens(blk_text)
        is_dup = False
        for prev_i, prev in enumerate(kept[-6:], start=max(0, len(kept) - 6)):
            prev_text = prev.content.strip()
            prev_tokens = _tokens(prev_text)
            prev_conf = prev.ocr_confidence or 0.0
            block_conf = block.ocr_confidence or 0.0

            # Case 1: high token-level similarity (near-exact duplicate)
            if _sim(blk_text, prev_text) >= 0.80:
                if block_conf > prev_conf + 0.05:
                    kept[prev_i] = block
                is_dup = True
                break

            # Case 2: shorter block's tokens are a subset of the longer block's tokens
            # (partial duplicate: one block captured only part of the other)
            if len(blk_tokens) >= 2 and len(prev_tokens) >= 2:
                blk_shorter = len(blk_tokens) < len(prev_tokens)
                shorter_t = blk_tokens if blk_shorter else prev_tokens
                longer_t = prev_tokens if blk_shorter else blk_tokens
                coverage = len(shorter_t & longer_t) / max(1, len(shorter_t))
                if coverage >= 0.85:
                    if blk_shorter:
                        # current is the subset → discard current
                        is_dup = True
                    else:
                        # current is the superset → replace prev with longer block
                        kept[prev_i] = block
                        is_dup = True
                    break

        if not is_dup:
            kept.append(block)
    return kept


def _merge_fragmented_lines(blocks: list[ParsedBlock], *, gap_px: float = 12.0) -> list[ParsedBlock]:
    """Merge blocks that are on the same horizontal line and close together.

    PaddleOCR sometimes splits one visual text line into 2-3 narrow boxes.
    This merges them back into a single block.
    """
    if not blocks:
        return blocks

    merged: list[ParsedBlock] = []
    current = blocks[0]

    for nxt in blocks[1:]:
        # Can only merge when both have bbox
        if current.bbox is None or nxt.bbox is None:
            merged.append(current)
            current = nxt
            continue

        # Same line: vertical overlap >= 50% AND horizontal gap is small
        cy_cur = (current.bbox.y1 + current.bbox.y2) / 2
        cy_nxt = (nxt.bbox.y1 + nxt.bbox.y2) / 2
        height_cur = max(1.0, current.bbox.y2 - current.bbox.y1)
        vertical_close = abs(cy_cur - cy_nxt) < height_cur * 0.6

        horizontal_gap = nxt.bbox.x1 - current.bbox.x2
        horizontal_ok = -5 <= horizontal_gap <= gap_px  # allow slight overlap too

        # Don't merge if they belong to different block types (e.g. heading vs paragraph)
        same_type = current.block_type == nxt.block_type

        if vertical_close and horizontal_ok and same_type:
            merged_content = current.content.rstrip() + " " + nxt.content.lstrip()
            merged_conf = (
                min(c for c in [current.ocr_confidence, nxt.ocr_confidence] if c is not None)
                if current.ocr_confidence is not None and nxt.ocr_confidence is not None
                else current.ocr_confidence
            )
            merged_bbox = BBox(
                x1=min(current.bbox.x1, nxt.bbox.x1),
                y1=min(current.bbox.y1, nxt.bbox.y1),
                x2=max(current.bbox.x2, nxt.bbox.x2),
                y2=max(current.bbox.y2, nxt.bbox.y2),
            )
            current = current.model_copy(update={
                "content": merged_content,
                "ocr_confidence": merged_conf,
                "bbox": merged_bbox,
            })
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


# ── Main OCR engine ───────────────────────────────────────────────────────────

class PaddleOCREngine:
    def __init__(self, *, lang: str = "vi", settings=None) -> None:
        self.lang = lang
        self.settings = settings
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            _configure_ocr_cache()
            if importlib.util.find_spec("paddle") is None:
                raise DependencyUnavailableError("paddlepaddle is required for PaddleOCR runtime")
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise DependencyUnavailableError("paddleocr is required for printed image OCR") from exc

            extra_kwargs: dict = {}
            if self.settings is not None:
                if self.settings.ocr_text_detection_model_name:
                    extra_kwargs["text_detection_model_name"] = self.settings.ocr_text_detection_model_name
                if getattr(self.settings, "ocr_text_recognition_model_name", None):
                    extra_kwargs["text_recognition_model_name"] = self.settings.ocr_text_recognition_model_name
                if self.settings.ocr_text_det_limit_side_len:
                    extra_kwargs["text_det_limit_side_len"] = self.settings.ocr_text_det_limit_side_len
                if self.settings.ocr_text_det_limit_type:
                    extra_kwargs["text_det_limit_type"] = self.settings.ocr_text_det_limit_type
                if self.settings.ocr_rec_score_threshold:
                    extra_kwargs["text_rec_score_thresh"] = self.settings.ocr_rec_score_threshold

            try:
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    device="cpu",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    **extra_kwargs,
                )
            except TypeError as exc:
                logger.warning("PaddleOCR rejected lightweight kwargs (%s); falling back to legacy init", exc)
                try:
                    self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True)
                except TypeError:
                    self._ocr = PaddleOCR(lang=self.lang)
        return self._ocr

    def parse_image(self, image_path: Path, *, language: str = "vi") -> ParsedDocument:
        try:
            blocks, variant_meta = self._run_enhanced_ocr(image_path, language=language)
        except Exception as exc:
            logger.exception(
                "PaddleOCR failed; no OCR fallback is allowed",
                extra={"image_path": str(image_path), "language": language},
            )
            raise DependencyUnavailableError(
                "PaddleOCR is required for OCR and no fallback OCR engine is allowed"
            ) from exc
        page_confidence_values = [block.ocr_confidence for block in blocks if block.ocr_confidence is not None]
        page_confidence = (
            sum(page_confidence_values) / len(page_confidence_values) if page_confidence_values else None
        )
        return ParsedDocument(
            source_path=str(image_path),
            file_type=image_path.suffix.lower().lstrip("."),
            language=language,
            pages=[ParsedPage(page_number=1, ocr_confidence=page_confidence, blocks=blocks)],
            extra={
                "parser": "paddleocr",
                "ocr_lang": self.lang,
                "det_model": getattr(self.settings, "ocr_text_detection_model_name", None),
                "rec_model": getattr(self.settings, "ocr_text_recognition_model_name", None),
                **variant_meta,
            },
        )

    def _run_enhanced_ocr(self, image_path: Path, *, language: str) -> tuple[list[ParsedBlock], dict]:
        """Multi-variant OCR: run on original + preprocessed variants, pick best per block."""
        # Run on original first
        primary_blocks = self._parse_blocks(self._run_ocr(image_path), image_path=image_path, language=language)
        variants_used: list[str] = ["original"]

        # Build preprocessed variants (enhanced, optionally binarized)
        cache_dir = _workspace_cache_dir("ocr_preprocess")
        preprocessed = _ImagePreprocessor.build_variants(image_path, cache_dir=cache_dir)

        if not preprocessed:
            # cv2 unavailable — fall back to grayscale-only path
            if not self._should_run_grayscale(primary_blocks):
                return self._finalize(primary_blocks), {
                    "ocr_preprocessing": "original_only",
                    "ocr_variants": variants_used,
                }
            grayscale_path = self._create_grayscale_variant(image_path)
            if grayscale_path:
                gray_blocks = self._parse_blocks(self._run_ocr(grayscale_path), image_path=image_path, language=language)
                primary_blocks = self._merge_variant_blocks(primary_blocks, gray_blocks, image_path=image_path, language=language, variant_name="grayscale")
                variants_used.append("grayscale")
            return self._finalize(primary_blocks), {
                "ocr_preprocessing": "grayscale_fallback",
                "ocr_variants": variants_used,
            }

        # Run OCR on each preprocessed variant and keep best result per block
        best_blocks = primary_blocks
        best_avg_conf = self._avg_confidence(primary_blocks)

        for variant_name, variant_path in preprocessed.items():
            try:
                variant_raw = self._run_ocr(variant_path)
                variant_blocks = self._parse_blocks(variant_raw, image_path=image_path, language=language)
            except Exception:
                logger.exception("OCR on variant '%s' failed", variant_name)
                continue

            avg_conf = self._avg_confidence(variant_blocks)
            logger.info(
                "OCR variant result",
                extra={
                    "variant": variant_name,
                    "blocks": len(variant_blocks),
                    "avg_confidence": round(avg_conf, 3),
                    "source": str(image_path.name),
                },
            )

            # Merge: replace base blocks where variant scores higher
            merged = self._merge_variant_blocks(
                best_blocks, variant_blocks,
                image_path=image_path,
                language=language,
                variant_name=variant_name,
            )
            merged_conf = self._avg_confidence(merged)
            if merged_conf >= best_avg_conf - 0.005:  # accept if not much worse
                best_blocks = merged
                best_avg_conf = merged_conf
            variants_used.append(variant_name)

        return self._finalize(best_blocks), {
            "ocr_preprocessing": "multi_variant",
            "ocr_variants": variants_used,
            "avg_confidence": round(best_avg_conf, 3),
        }

    def _finalize(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        """Sort by reading order, merge fragmented lines, deduplicate."""
        sorted_blocks = _sort_blocks_by_reading_order(blocks)
        merged = _merge_fragmented_lines(sorted_blocks)
        deduped = _deduplicate_blocks(merged)
        # Re-index reading_order after sort/merge/dedup
        return [
            b.model_copy(update={"block_index": i, "reading_order": i})
            for i, b in enumerate(deduped)
        ]

    def _parse_blocks(self, raw_result, *, image_path: Path, language: str) -> list[ParsedBlock]:
        """Parse raw OCR output and apply text normalization."""
        blocks = self._parse_result(raw_result, image_path=image_path, language=language)
        # Apply character-level normalization for Vietnamese lookalikes
        normalized = []
        for block in blocks:
            fixed = _normalize_vi_text(block.content)
            if fixed != block.content:
                block = block.model_copy(update={"content": fixed})
            normalized.append(block)
        return normalized

    @staticmethod
    def _avg_confidence(blocks: list[ParsedBlock]) -> float:
        scores = [b.ocr_confidence for b in blocks if b.ocr_confidence is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def _should_run_grayscale(self, primary_blocks: list[ParsedBlock]) -> bool:
        mode = "auto"
        trigger = 0.85
        if self.settings is not None:
            mode = (self.settings.ocr_enable_grayscale_variant or "auto").lower()
            trigger = self.settings.ocr_grayscale_trigger_confidence
        if mode in {"true", "yes", "1", "always"}:
            return True
        if mode in {"false", "no", "0", "never"}:
            return False
        confidences = [block.ocr_confidence for block in primary_blocks if block.ocr_confidence is not None]
        if not confidences:
            return True
        avg = sum(confidences) / len(confidences)
        return avg < trigger

    def _create_grayscale_variant(self, image_path: Path) -> Path | None:
        cv2_spec = importlib.util.find_spec("cv2")
        if cv2_spec is None:
            return None
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            return None
        digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:16]
        output_dir = _workspace_cache_dir("ocr_preprocess")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}-{digest}-grayscale.png"
        if output_path.exists():
            return output_path
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(output_path), gray_bgr)
        return output_path

    def _merge_variant_blocks(
        self,
        base_blocks: list[ParsedBlock],
        variant_blocks: list[ParsedBlock],
        *,
        image_path: Path,
        language: str,
        variant_name: str,
    ) -> list[ParsedBlock]:
        if not base_blocks:
            return [
                block.model_copy(update={"extra": {**block.extra, "ocr_source_variant": variant_name}})
                for block in variant_blocks
            ]
        merged: list[ParsedBlock] = []
        used_variant_indexes: set[int] = set()
        for index, base in enumerate(base_blocks):
            match_index = self._find_matching_block(base, variant_blocks, used_variant_indexes, fallback_index=index)
            if match_index is None:
                merged.append(base)
                continue
            used_variant_indexes.add(match_index)
            candidate = variant_blocks[match_index]
            if self._should_replace_block(base, candidate, language=language):
                merged.append(
                    base.model_copy(
                        update={
                            "block_id": f"blk-{uuid5(NAMESPACE_URL, f'{image_path}:paddleocr:1:{base.block_index}:{candidate.content}').hex[:12]}",
                            "content": candidate.content,
                            "ocr_confidence": candidate.ocr_confidence,
                            "bbox": candidate.bbox or base.bbox,
                            "extra": {
                                **base.extra,
                                "ocr_source_variant": variant_name,
                                "ocr_raw_content": base.content,
                                "ocr_raw_confidence": base.ocr_confidence,
                            },
                        }
                    )
                )
            else:
                merged.append(base)
        return merged

    def _find_matching_block(
        self,
        base: ParsedBlock,
        candidates: list[ParsedBlock],
        used_indexes: set[int],
        *,
        fallback_index: int,
    ) -> int | None:
        if base.bbox is None:
            # Without bbox: only use fallback when list sizes match (same page structure).
            # Avoids wrong assignments when variants detect different block counts.
            if len(candidates) == fallback_index + 1 or (
                fallback_index < len(candidates) and fallback_index not in used_indexes
            ):
                return fallback_index if fallback_index < len(candidates) and fallback_index not in used_indexes else None
            return None
        best_index = None
        best_overlap = 0.0
        for index, candidate in enumerate(candidates):
            if index in used_indexes or candidate.bbox is None:
                continue
            overlap = self._iou_overlap(base.bbox, candidate.bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        # Only accept when there's a meaningful spatial match (≥ 40% IoU).
        # Stricter than before to prevent wrong variant blocks from replacing base.
        if best_overlap >= 0.40:
            return best_index
        return None

    @staticmethod
    def _vertical_overlap(first: BBox, second: BBox) -> float:
        top = max(first.y1, second.y1)
        bottom = min(first.y2, second.y2)
        overlap = max(0.0, bottom - top)
        first_height = max(1.0, first.y2 - first.y1)
        second_height = max(1.0, second.y2 - second.y1)
        return overlap / min(first_height, second_height)

    @staticmethod
    def _iou_overlap(first: BBox, second: BBox) -> float:
        """Intersection-over-Union for bbox matching across variants."""
        ix1 = max(first.x1, second.x1)
        iy1 = max(first.y1, second.y1)
        ix2 = min(first.x2, second.x2)
        iy2 = min(first.y2, second.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area1 = max(1.0, (first.x2 - first.x1) * (first.y2 - first.y1))
        area2 = max(1.0, (second.x2 - second.x1) * (second.y2 - second.y1))
        return inter / (area1 + area2 - inter)

    def _should_replace_block(self, base: ParsedBlock, candidate: ParsedBlock, *, language: str) -> bool:
        if not candidate.content.strip():
            return False
        if candidate.content == base.content:
            return False
        base_confidence = base.ocr_confidence or 0.0
        candidate_confidence = candidate.ocr_confidence or 0.0
        if candidate_confidence + 0.12 < base_confidence:
            return False
        return self._text_quality_score(candidate.content, candidate_confidence, language=language) > (
            self._text_quality_score(base.content, base_confidence, language=language) + 0.15
        )

    @staticmethod
    def _text_quality_score(text: str, confidence: float, *, language: str) -> float:
        cleaned = text.strip()
        score = confidence * 10.0
        if language == "vi":
            accent_count = sum(_has_vietnamese_mark(char) for char in cleaned)
            suspect_count = sum(char in _SUSPECT_LATIN_DIACRITICS for char in cleaned)
            score += min(accent_count, 20) * 0.45
            score -= suspect_count * 1.5
        if len(cleaned) == 1 and cleaned.isalpha():
            score -= 1.0
        return score

    def _run_ocr(self, image_path: Path):
        predict = getattr(self.ocr, "predict", None)
        if callable(predict):
            return predict(str(image_path))
        return self.ocr.ocr(str(image_path), cls=True)

    def _parse_result(self, raw_result, *, image_path: Path, language: str) -> list[ParsedBlock]:
        structured = self._parse_structured_result(raw_result, image_path=image_path, language=language)
        if structured:
            return structured
        lines = raw_result[0] if raw_result and isinstance(raw_result[0], list) else raw_result
        blocks: list[ParsedBlock] = []
        for index, line in enumerate(lines or []):
            parsed = self._parse_line(line)
            if parsed is None:
                continue
            bbox, text, confidence = parsed
            blocks.append(
                ParsedBlock(
                    block_id=f"blk-{uuid5(NAMESPACE_URL, f'{image_path}:1:{index}:{text}').hex[:12]}",
                    block_index=len(blocks),
                    block_type=BlockType.OCR_TEXT.value,
                    content=text,
                    page_number=1,
                    language=language,
                    bbox=bbox,
                    ocr_confidence=confidence,
                    reading_order=len(blocks),
                    source="paddleocr",
                )
            )
        return blocks

    def _parse_structured_result(self, raw_result, *, image_path: Path, language: str) -> list[ParsedBlock]:
        result_items = raw_result if isinstance(raw_result, list) else [raw_result]
        blocks: list[ParsedBlock] = []
        for item in result_items:
            data = self._result_to_dict(item)
            if not data:
                continue
            texts = data.get("rec_texts") or data.get("texts") or []
            scores = data.get("rec_scores") or data.get("scores") or []
            boxes = data.get("rec_polys") or data.get("dt_polys") or data.get("boxes") or []
            for index, text in enumerate(texts):
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                confidence = float(scores[index]) if index < len(scores) and scores[index] is not None else None
                bbox = self._bbox_from_polygon(boxes[index]) if index < len(boxes) else None
                blocks.append(
                    ParsedBlock(
                        block_id=f"blk-{uuid5(NAMESPACE_URL, f'{image_path}:paddleocr:1:{len(blocks)}:{cleaned}').hex[:12]}",
                        block_index=len(blocks),
                        block_type=BlockType.OCR_TEXT.value,
                        content=cleaned,
                        page_number=1,
                        language=language,
                        bbox=bbox,
                        ocr_confidence=confidence,
                        reading_order=len(blocks),
                        source="paddleocr",
                    )
                )
        return blocks

    @staticmethod
    def _result_to_dict(item) -> dict | None:
        if isinstance(item, dict):
            return item
        for attr in ("json", "res"):
            value = getattr(item, attr, None)
            if isinstance(value, dict):
                return value.get("res", value)
        method = getattr(item, "to_json", None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return value.get("res", value)
        return None

    @staticmethod
    def _bbox_from_polygon(polygon) -> BBox | None:
        try:
            points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
            coordinates = [point.tolist() if hasattr(point, "tolist") else point for point in points]
            xs = [float(point[0]) for point in coordinates if isinstance(point, (list, tuple)) and len(point) >= 2]
            ys = [float(point[1]) for point in coordinates if isinstance(point, (list, tuple)) and len(point) >= 2]
        except TypeError:
            return None
        if not xs or not ys:
            return None
        return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

    @staticmethod
    def _parse_line(line) -> tuple[BBox | None, str, float | None] | None:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            return None
        polygon = line[0]
        text_info = line[1]
        if not isinstance(text_info, (list, tuple)) or not text_info:
            return None
        text = str(text_info[0]).strip()
        if not text:
            return None
        confidence = float(text_info[1]) if len(text_info) > 1 and text_info[1] is not None else None
        bbox = None
        if isinstance(polygon, (list, tuple)) and polygon:
            xs = [float(point[0]) for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
            ys = [float(point[1]) for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
            if xs and ys:
                bbox = BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
        return bbox, text, confidence


# ── EasyOCR engine (Vietnamese primary) ──────────────────────────────────────

class EasyOCREngine:
    """OCR engine backed by EasyOCR — primary choice for Vietnamese images.

    EasyOCR ships its own Vietnamese recognition model trained on real
    Vietnamese text (with tone marks), so it avoids the tone-mark loss
    that PaddleOCR v3's Latin model suffers from.
    """

    def __init__(self, *, lang: str = "vi", gpu: bool = False) -> None:
        self.lang = lang
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise DependencyUnavailableError("easyocr is required for Vietnamese OCR") from exc
            # Map our lang codes to EasyOCR language list
            lang_list = ["vi"] if self.lang == "vi" else ["en"]
            self._reader = easyocr.Reader(lang_list, gpu=self.gpu, verbose=False)
        return self._reader

    def parse_image(self, image_path: Path, *, language: str = "vi") -> ParsedDocument:
        blocks, meta = self._run_ocr_with_preprocessing(image_path, language=language)
        page_confidence_values = [b.ocr_confidence for b in blocks if b.ocr_confidence is not None]
        page_confidence = (
            sum(page_confidence_values) / len(page_confidence_values) if page_confidence_values else None
        )
        return ParsedDocument(
            source_path=str(image_path),
            file_type=image_path.suffix.lower().lstrip("."),
            language=language,
            pages=[ParsedPage(page_number=1, ocr_confidence=page_confidence, blocks=blocks)],
            extra={"parser": "easyocr", "ocr_lang": self.lang, **meta},
        )

    def _run_ocr_with_preprocessing(self, image_path: Path, *, language: str) -> tuple[list[ParsedBlock], dict]:
        primary = self._ocr_blocks(image_path, language=language)

        cache_dir = _workspace_cache_dir("ocr_preprocess")
        preprocessed = _ImagePreprocessor.build_variants(image_path, cache_dir=cache_dir)

        if not preprocessed:
            return self._finalize(primary), {"ocr_preprocessing": "original_only"}

        best = primary
        best_conf = self._avg_confidence(primary)
        variants_used = ["original"]

        for variant_name, variant_path in preprocessed.items():
            try:
                variant_blocks = self._ocr_blocks(variant_path, language=language)
            except Exception:
                logger.exception("EasyOCR on variant '%s' failed", variant_name)
                continue
            avg = self._avg_confidence(variant_blocks)
            logger.info(
                "EasyOCR variant result",
                extra={"variant": variant_name, "blocks": len(variant_blocks), "avg_confidence": round(avg, 3)},
            )
            if avg > best_conf:
                best = variant_blocks
                best_conf = avg
            variants_used.append(variant_name)

        return self._finalize(best), {
            "ocr_preprocessing": "multi_variant",
            "ocr_variants": variants_used,
            "avg_confidence": round(best_conf, 3),
        }

    def _ocr_blocks(self, image_path: Path, *, language: str) -> list[ParsedBlock]:
        raw = self.reader.readtext(str(image_path), detail=1, paragraph=False)
        blocks: list[ParsedBlock] = []
        for idx, item in enumerate(raw or []):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            polygon, text, confidence = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue
            bbox = self._bbox_from_polygon(polygon)
            blocks.append(
                ParsedBlock(
                    block_id=f"blk-{uuid5(NAMESPACE_URL, f'{image_path}:easyocr:1:{idx}:{text}').hex[:12]}",
                    block_index=len(blocks),
                    block_type=BlockType.OCR_TEXT.value,
                    content=text,
                    page_number=1,
                    language=language,
                    bbox=bbox,
                    ocr_confidence=confidence,
                    reading_order=len(blocks),
                    source="easyocr",
                )
            )
        return blocks

    def _finalize(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        sorted_blocks = _sort_blocks_by_reading_order(blocks)
        merged = _merge_fragmented_lines(sorted_blocks)
        deduped = _deduplicate_blocks(merged)
        return [
            b.model_copy(update={"block_index": i, "reading_order": i})
            for i, b in enumerate(deduped)
        ]

    @staticmethod
    def _avg_confidence(blocks: list[ParsedBlock]) -> float:
        scores = [b.ocr_confidence for b in blocks if b.ocr_confidence is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _bbox_from_polygon(polygon) -> BBox | None:
        try:
            points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
            xs = [float(p[0]) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [float(p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
        except (TypeError, IndexError):
            return None
        if not xs or not ys:
            return None
        return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
