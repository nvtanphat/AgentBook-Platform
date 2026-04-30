from __future__ import annotations

import re
from collections import OrderedDict

from src.processing.types import ExtractedEntity


class EntityResolver:
    def resolve(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        resolved: OrderedDict[str, ExtractedEntity] = OrderedDict()
        for entity in entities:
            key = self.normalize(entity.canonical_name)
            existing = resolved.get(key)
            if existing is None:
                resolved[key] = entity.model_copy(update={"aliases": sorted(set(entity.aliases + [entity.canonical_name]))})
                continue
            resolved[key] = existing.model_copy(
                update={
                    "aliases": sorted(set(existing.aliases + entity.aliases + [entity.canonical_name])),
                    "mention_refs": existing.mention_refs + entity.mention_refs,
                    "confidence": max(existing.confidence, entity.confidence),
                }
            )
        return list(resolved.values())

    @staticmethod
    def normalize(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        aliases = {"dropout regularization": "dropout", "dropout layer": "dropout", "l 1": "l1", "l 2": "l2"}
        return aliases.get(normalized, normalized)
