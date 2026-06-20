from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPoint:
    score: float
    is_relevant: bool


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_accept_rate: float
    false_refusal_rate: float


class ThresholdCalibrator:
    def calibrate(self, points: list[CalibrationPoint], thresholds: list[float]) -> ThresholdReport:
        if not points:
            return ThresholdReport(threshold=0.55, precision=0.0, recall=0.0, f1=0.0, false_accept_rate=0.0, false_refusal_rate=0.0)
        reports = [self.evaluate(points, threshold) for threshold in thresholds]
        return max(reports, key=lambda report: (report.f1, report.precision, report.recall))

    @staticmethod
    def evaluate(points: list[CalibrationPoint], threshold: float) -> ThresholdReport:
        tp = fp = tn = fn = 0
        for point in points:
            accepted = point.score >= threshold
            if accepted and point.is_relevant:
                tp += 1
            elif accepted and not point.is_relevant:
                fp += 1
            elif not accepted and point.is_relevant:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        false_accept_rate = fp / (fp + tn) if fp + tn else 0.0
        false_refusal_rate = fn / (fn + tp) if fn + tp else 0.0
        return ThresholdReport(
            threshold=threshold,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            false_accept_rate=round(false_accept_rate, 4),
            false_refusal_rate=round(false_refusal_rate, 4),
        )
