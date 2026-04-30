from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.config import Settings


@dataclass(frozen=True)
class ImageQualityReport:
    score: float
    is_acceptable: bool
    blur_variance: float
    brightness: float
    contrast: float
    skew_degrees: float
    warnings: list[str]


class ImageQualityChecker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check(self, image_path: Path) -> ImageQualityReport:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "opencv-python (cv2) and numpy are required for image quality checking"
            ) from exc

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image file: {image_path}")

        blur_variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
        brightness = float(np.mean(image))
        contrast = float(np.std(image))
        skew_degrees = self._estimate_skew(image)

        warnings: list[str] = []
        score_parts: list[float] = []

        score_parts.append(min(1.0, blur_variance / self.settings.min_blur_variance))
        if blur_variance < self.settings.min_blur_variance:
            warnings.append("image is too blurry for reliable handwriting evidence")

        if brightness < self.settings.min_brightness:
            warnings.append("image is too dark for reliable handwriting evidence")
        if brightness > self.settings.max_brightness:
            warnings.append("image is too bright for reliable handwriting evidence")
        brightness_score = 1.0 if self.settings.min_brightness <= brightness <= self.settings.max_brightness else 0.45
        score_parts.append(brightness_score)

        score_parts.append(min(1.0, contrast / self.settings.min_contrast))
        if contrast < self.settings.min_contrast:
            warnings.append("image contrast is too low for reliable handwriting evidence")

        skew_abs = abs(skew_degrees)
        score_parts.append(max(0.0, 1.0 - (skew_abs / max(self.settings.max_abs_skew_degrees, 1.0))))
        if skew_abs > self.settings.max_abs_skew_degrees:
            warnings.append("image is too skewed for reliable handwriting evidence")

        score = round(sum(score_parts) / len(score_parts), 4)
        is_acceptable = score >= self.settings.min_handwriting_quality_score and not warnings
        return ImageQualityReport(
            score=score,
            is_acceptable=is_acceptable,
            blur_variance=blur_variance,
            brightness=brightness,
            contrast=contrast,
            skew_degrees=skew_degrees,
            warnings=warnings,
        )

    @staticmethod
    def _estimate_skew(image) -> float:
        import cv2
        import numpy as np

        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 20:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if angle > 45:
            angle = angle - 90
        return float(angle)
