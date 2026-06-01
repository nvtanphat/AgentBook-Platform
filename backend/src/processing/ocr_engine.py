from __future__ import annotations

import hashlib
import logging
import os
import unicodedata
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from src.processing.types import BBox, BlockType, DependencyUnavailableError, ParsedBlock, ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)


_VIETNAMESE_SPECIALS = {chr(codepoint) for codepoint in (0x0111, 0x0110, 0x01A1, 0x01A0, 0x01B0, 0x01AF)}
_SUSPECT_LATIN_DIACRITICS = {chr(codepoint) for codepoint in (0x0113, 0x016B, 0x01CE, 0x00E5, 0x012B, 0x014D)}

# Latin Extended lookalikes that OCR engines commonly confuse with Vietnamese chars.
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
        "HF_HOME": _workspace_cache_dir("huggingface"),
        "MODELSCOPE_CACHE": _workspace_cache_dir("modelscope"),
    }
    for name, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


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
    """Fix Latin Extended lookalikes that OCR engines substitute for Vietnamese chars."""
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


# ── VietOCR recognizer (Vietnamese tone-accurate recognition) ─────────────────

class VietOCRRecognizer:
    """Transformer-based Vietnamese text-line recognizer.

    EasyOCR's detection (box finding) is solid but its recognition drops
    Vietnamese tone marks on scanned text. VietOCR (vgg_transformer, trained
    on Vietnamese) reads diacritics far better. We keep EasyOCR for detection
    and swap in VietOCR for per-box recognition.

    Recognition-only: needs a cropped single text-line image, returns the
    decoded string. Lazy-loads the predictor on first use.
    """

    def __init__(self, *, device: str = "cpu", model_name: str = "vgg_transformer") -> None:
        self.device = device
        self.model_name = model_name
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            try:
                from vietocr.tool.config import Cfg
                from vietocr.tool.predictor import Predictor
            except ImportError as exc:
                raise DependencyUnavailableError("vietocr is required for VietOCR recognition") from exc
            cfg = Cfg.load_config_from_name(self.model_name)
            cfg["device"] = self.device
            # Avoid re-downloading the CNN backbone; the seq model weights are enough.
            cfg["cnn"]["pretrained"] = False
            self._predictor = Predictor(cfg)
        return self._predictor

    def predict(self, pil_image) -> str:
        """Recognize a single cropped text-line PIL image. Returns '' on failure."""
        try:
            text = self.predictor.predict(pil_image)
        except Exception:
            logger.exception("VietOCR recognition failed on a crop")
            return ""
        return (text or "").strip()


# ── EasyOCR engine (Vietnamese primary) ──────────────────────────────────────

class EasyOCREngine:
    """OCR engine backed by EasyOCR for detection + recognition.

    EasyOCR ships its own Vietnamese recognition model, but on scanned text it
    loses tone marks. When a `recognizer` (e.g. VietOCRRecognizer) is attached,
    EasyOCR is used only for box DETECTION and each box is re-recognized by the
    recognizer — best-of-breed: EasyOCR detection + VietOCR Vietnamese reading.
    """

    def __init__(self, *, lang: str = "vi", gpu: bool = False, recognizer: "VietOCRRecognizer | None" = None) -> None:
        self.lang = lang
        self.gpu = gpu
        self.recognizer = recognizer
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
        # Variant selection uses EasyOCR confidence only (recognizer OFF) — cheap.
        # The VietOCR recognition pass (expensive) runs ONCE on the winning variant.
        primary = self._ocr_blocks(image_path, language=language, apply_recognizer=False)

        cache_dir = _workspace_cache_dir("ocr_preprocess")
        preprocessed = _ImagePreprocessor.build_variants(image_path, cache_dir=cache_dir)

        best_path = image_path
        best = primary
        best_conf = self._avg_confidence(primary)
        variants_used = ["original"]

        for variant_name, variant_path in (preprocessed or {}).items():
            try:
                variant_blocks = self._ocr_blocks(variant_path, language=language, apply_recognizer=False)
            except Exception:
                logger.exception("EasyOCR on variant '%s' failed", variant_name)
                continue
            avg = self._avg_confidence(variant_blocks)
            logger.info(
                "EasyOCR variant result",
                extra={"variant": variant_name, "blocks": len(variant_blocks), "avg_confidence": round(avg, 3)},
            )
            if avg > best_conf:
                best, best_conf, best_path = variant_blocks, avg, variant_path
            variants_used.append(variant_name)

        # Single recognition pass on the winning variant when a recognizer is set.
        if self.recognizer is not None:
            best = self._ocr_blocks(best_path, language=language, apply_recognizer=True)

        meta = (
            {"ocr_preprocessing": "original_only"}
            if not preprocessed
            else {"ocr_preprocessing": "multi_variant", "ocr_variants": variants_used, "avg_confidence": round(best_conf, 3)}
        )
        return self._finalize(best), meta

    def _ocr_blocks(self, image_path: Path, *, language: str, apply_recognizer: bool = True) -> list[ParsedBlock]:
        raw = self.reader.readtext(str(image_path), detail=1, paragraph=False)
        # When a recognizer is attached, re-read each detected box from the
        # source image — EasyOCR boxes, recognizer text (better VN diacritics).
        source_image = None
        if apply_recognizer and self.recognizer is not None and raw:
            try:
                from PIL import Image
                source_image = Image.open(str(image_path)).convert("RGB")
            except Exception:
                logger.exception("Failed to open image for recognizer; using EasyOCR text")
                source_image = None
        blocks: list[ParsedBlock] = []
        for idx, item in enumerate(raw or []):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            polygon, text, confidence = item[0], str(item[1]).strip(), float(item[2])
            bbox = self._bbox_from_polygon(polygon)
            if source_image is not None and bbox is not None:
                recognized = self._recognize_crop(source_image, bbox)
                if recognized:
                    text, confidence = recognized, max(confidence, 0.5)
            if not text:
                continue
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

    def _recognize_crop(self, source_image, bbox: BBox) -> str:
        """Crop the bbox region (with small padding) and recognize via VietOCR."""
        if self.recognizer is None:
            return ""
        w, h = source_image.size
        pad = 2
        left = max(0, int(bbox.x1) - pad)
        top = max(0, int(bbox.y1) - pad)
        right = min(w, int(bbox.x2) + pad)
        bottom = min(h, int(bbox.y2) + pad)
        if right <= left or bottom <= top:
            return ""
        crop = source_image.crop((left, top, right, bottom))
        return self.recognizer.predict(crop)

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
