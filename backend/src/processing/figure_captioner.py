"""VLM-based figure captioner for charts, diagrams, and images in documents.

Flow:
  1. Receive a figure image (path or cropped region bytes).
  2. Try a local vision-capable Ollama model (qwen2-vl or llava).
  3. Fall back to PaddleOCR if VLM is unavailable / returns empty.
  4. Return a plain-text caption to be stored as the FIGURE block content.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import re
import tempfile
import threading
from pathlib import Path
from typing import Protocol

from src.processing.types import BBox

logger = logging.getLogger(__name__)

# Prompt used for VLM captioning. Short and structured so the model stays focused.
_CAPTION_PROMPT_VI = (
    "Read all text visible in this image. "
    "List every label, term, number, formula, and step you can see. "
    "Respond in Vietnamese. Format as a structured list. "
    "Only report what is actually shown, do not infer."
)
_CAPTION_PROMPT_EN = (
    "Read all text visible in this image. "
    "List every label, term, number, formula, and step you can see. "
    "Format as a structured list. "
    "Only report what is actually shown, do not infer."
)

# Vision models supported by Ollama, tried in order.
_OLLAMA_VISION_MODELS = ["minicpm-v", "qwen2-vl", "llava", "moondream", "bakllava", "llava-phi3"]


class FigureCaptioner:
    """Caption document figures using a local VLM with OCR fallback."""

    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434",
        language: str = "vi",
        ocr_fallback: bool = True,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.language = language
        self.ocr_fallback = ocr_fallback
        self._available_model: str | None = None
        self._model_checked: bool = False
        self._ocr_engine = None  # lazy-init, reused across all figures
        self._ocr_engine_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def caption_image_path(self, image_path: Path) -> str:
        """Return a caption for a standalone image file."""
        return self._caption(image_path=image_path)

    def caption_page_region(
        self,
        page_image_path: Path,
        bbox: BBox,
        *,
        page_width: int,
        page_height: int,
    ) -> str:
        """Crop a bbox region from a page image and return a caption.

        bbox coordinates are in Docling's page coordinate system (points).
        """
        cropped = self._crop_region(page_image_path, bbox, page_width=page_width, page_height=page_height)
        if cropped is None:
            return ""
        return self._caption(image_path=cropped)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _caption(self, *, image_path: Path) -> str:
        if not image_path.exists() or image_path.stat().st_size < 512:
            return ""

        # Try VLM first
        vlm_caption = self._try_vlm(image_path)
        if vlm_caption:
            return vlm_caption

        # Fall back to OCR (extracts any text that appears in the figure)
        if self.ocr_fallback:
            return self._try_ocr_fallback(image_path)

        return ""

    def _try_vlm(self, image_path: Path) -> str:
        model = self._detect_available_model()
        if model is None:
            return ""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — skipping VLM captioning")
            return ""

        prompt = _CAPTION_PROMPT_VI if self.language == "vi" else _CAPTION_PROMPT_EN
        image_b64 = self._encode_image_resized(image_path)

        try:
            response = httpx.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            caption = (data.get("response") or "").strip()
            caption = self._remove_repetition_loops(caption)
            if self._looks_like_cross_language_hallucination(caption):
                logger.warning(
                    "VLM caption rejected due to cross-language hallucination",
                    extra={"model": model, "image": image_path.name, "chars": len(caption)},
                )
                return ""
            if caption:
                logger.info(
                    "Figure captioned via VLM",
                    extra={"model": model, "image": image_path.name, "chars": len(caption)},
                )
            return caption
        except Exception as exc:
            logger.warning(
                "VLM captioning failed",
                extra={"model": model, "error": str(exc), "error_type": type(exc).__name__},
            )
            return ""

    def _looks_like_cross_language_hallucination(self, text: str) -> bool:
        """Reject VLM captions that inject unrelated CJK text into vi/en OCR output."""
        if not text or self.language not in {"vi", "en"}:
            return False

        cjk_chars = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
        if not cjk_chars:
            return False

        visible_chars = [ch for ch in text if not ch.isspace()]
        cjk_ratio = len(cjk_chars) / max(1, len(visible_chars))

        # A genuine Vietnamese/English slide should not contain sustained CJK runs.
        # Short isolated symbols are tolerated, but VLM hallucinations often contain
        # repeated mixed fragments such as "đ用力導決策".
        has_cjk_run = re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}", text) is not None
        has_mixed_garble = re.search(r"[A-Za-zÀ-ỹ][\u3400-\u4dbf\u4e00-\u9fff]|[\u3400-\u4dbf\u4e00-\u9fff][A-Za-zÀ-ỹ]", text) is not None
        return cjk_ratio >= 0.02 or has_cjk_run or has_mixed_garble

    def _detect_available_model(self) -> str | None:
        if self._model_checked:
            return self._available_model
        self._model_checked = True
        try:
            import httpx
            resp = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            all_models = resp.json().get("models", [])
            installed = {m["name"].split(":")[0]: m["name"] for m in all_models}
            for model in _OLLAMA_VISION_MODELS:
                if model in installed:
                    self._available_model = installed[model]
                    logger.info("Figure captioner using VLM model: %s", self._available_model)
                    return self._available_model
        except Exception as exc:
            logger.debug("Ollama not reachable for figure captioning: %s", exc)
        logger.info("No vision model found in Ollama — figure captioner will use OCR fallback")
        return None

    @staticmethod
    def _encode_image_resized(image_path: Path, max_side: int = 1024) -> str:
        """Resize image to max_side px on longest side, return base64. Falls back to raw bytes."""
        try:
            from PIL import Image
            import io
            img = Image.open(image_path)
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return base64.b64encode(image_path.read_bytes()).decode("ascii")

    @staticmethod
    def _remove_repetition_loops(text: str) -> str:
        """Detect and truncate repeating token loops from VLM hallucination."""
        if not text:
            return text
        # Find repeating substrings >= 8 chars that appear 4+ times consecutively
        import re
        pattern = re.compile(r"(.{8,}?)\1{3,}", re.DOTALL)
        m = pattern.search(text)
        if m:
            text = text[:m.start()].rstrip(" /,\n") + "..."
        return text

    def _try_ocr_fallback(self, image_path: Path) -> str:
        """Run EasyOCR on the figure region to extract any embedded text."""
        try:
            if self._ocr_engine is None:
                with self._ocr_engine_lock:
                    if self._ocr_engine is None:
                        from src.processing.ocr_engine import EasyOCREngine
                        self._ocr_engine = EasyOCREngine(lang="vi" if self.language == "vi" else "en")
            parsed = self._ocr_engine.parse_image(image_path, language=self.language)
            text = " ".join(b.content for b in parsed.blocks if b.content.strip())
            if text:
                prefix = "[Hình - văn bản trích xuất]" if self.language == "vi" else "[Figure - extracted text]"
                return f"{prefix}: {text}"
        except Exception as exc:
            logger.debug("OCR fallback for figure failed: %s", exc)
        return ""

    @staticmethod
    def _crop_region(
        page_image_path: Path,
        bbox: BBox,
        *,
        page_width: int,
        page_height: int,
    ) -> Path | None:
        """Crop a region from a page image using Docling bbox (point coordinates)."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.debug("cv2 not available — cannot crop figure region")
            return None

        img = cv2.imread(str(page_image_path))
        if img is None:
            return None

        img_h, img_w = img.shape[:2]
        pw = max(1, page_width)
        ph = max(1, page_height)

        # Docling bbox: l/t/r/b in page points (origin = bottom-left).
        # Convert to pixel coordinates (origin = top-left).
        px1 = int(bbox.x1 / pw * img_w)
        px2 = int(bbox.x2 / pw * img_w)
        py1 = int((ph - bbox.y2) / ph * img_h)  # flip y-axis
        py2 = int((ph - bbox.y1) / ph * img_h)

        # Clamp to image bounds
        px1, px2 = max(0, px1), min(img_w, px2)
        py1, py2 = max(0, py1), min(img_h, py2)

        if px2 - px1 < 10 or py2 - py1 < 10:
            return None

        cropped = img[py1:py2, px1:px2]
        digest = hashlib.sha1(f"{page_image_path}:{bbox.x1:.1f},{bbox.y1:.1f}".encode()).hexdigest()[:12]
        out_dir = page_image_path.parent / "figure_crops"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fig-{digest}.png"
        cv2.imwrite(str(out_path), cropped)
        return out_path
