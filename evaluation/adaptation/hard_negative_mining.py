from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextDocument:
    text: str
    metadata: dict


class HardNegativeMiner:
    def mine(self, *, query: str, positive: str, candidates: list[TextDocument], limit: int) -> list[TextDocument]:
        query_terms = self._terms(query)
        positive_terms = self._terms(positive)
        scored: list[tuple[float, TextDocument]] = []
        for candidate in candidates:
            if candidate.text == positive:
                continue
            candidate_terms = self._terms(candidate.text)
            lexical_overlap = self._jaccard(query_terms, candidate_terms)
            answer_overlap = self._jaccard(positive_terms, candidate_terms)
            score = lexical_overlap - (answer_overlap * 0.35)
            if lexical_overlap > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored[:limit]]

    @staticmethod
    def _terms(text: str) -> set[str]:
        stopwords = {"the", "and", "for", "with", "from", "this", "that", "cua", "của", "trong", "nhung", "những"}
        return {token.lower() for token in re.findall(r"[\w\-]{3,}", text, flags=re.UNICODE) if token.lower() not in stopwords}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
