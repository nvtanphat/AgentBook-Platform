from __future__ import annotations

_overrides: dict[str, object] = {}


def get_override(key: str, default: object) -> object:
    return _overrides.get(key, default)


def set_override(key: str, value: object) -> None:
    _overrides[key] = value


def get_all_overrides() -> dict[str, object]:
    return dict(_overrides)
