from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    language: str = Field(pattern=r"^(vi|en)$")
    translated_query: str | None = None
    paraphrases: list[str] = Field(default_factory=list)


_REWRITE_PROMPT = """You are a query rewriter for a multilingual RAG system over English and Vietnamese learning materials.

Given a user query, output a JSON object with these fields:
- "language": "vi" if the query is Vietnamese, "en" if English
- "translated_query": If language=="vi", a faithful English translation preserving technical terms; if language=="en", null
- "paraphrases": exactly 2 short English paraphrases (each under 12 words) that use concrete domain vocabulary likely to appear verbatim in source documents — synonyms, expanded acronyms, related technical terms, or narrower/broader framings of the same concept. Avoid generic phrases; prefer terms a textbook author would use.

Rules:
- Output ONLY the JSON object — no prose, no markdown fences, no code blocks.
- Paraphrases must preserve the original intent. Do not invent facts or shift topic.
- Keep technical/proper nouns unchanged across languages.
- If the query already contains concrete domain terms, vary the surface form in the paraphrases rather than repeating them.

Query: {query}

JSON:"""


class LLMQueryRewriter:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def rewrite(self, query: str) -> RewriteResult | None:
        prompt = _REWRITE_PROMPT.format(query=query)
        try:
            raw = await self.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.warning("Query rewriter LLM call failed", extra={"error": str(exc)})
            return None

        parsed = self._parse_json(raw)
        if parsed is None:
            logger.warning("Query rewriter produced unparseable output", extra={"raw_preview": raw[:200]})
            return None

        try:
            result = RewriteResult.model_validate(parsed)
        except ValidationError as exc:
            logger.warning("Query rewriter output failed schema validation", extra={"error": str(exc)})
            return None

        # Sanitize paraphrases: drop empties, dedupe, cap at 3
        cleaned = []
        seen: set[str] = set()
        for paraphrase in result.paraphrases:
            text = paraphrase.strip()
            key = text.lower()
            if not text or key in seen or key == query.lower():
                continue
            seen.add(key)
            cleaned.append(text)
            if len(cleaned) >= 3:
                break
        result.paraphrases = cleaned
        return result

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        # Strip markdown fences if the model added them despite instructions
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Some models prepend prose — try to locate the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data
