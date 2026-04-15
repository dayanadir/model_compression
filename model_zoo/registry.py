"""Family registry: maps family names to ModelFamily subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_zoo.families.base import ModelFamily

_FAMILIES: dict[str, type[ModelFamily]] = {}


def register_family(cls: type[ModelFamily]) -> type[ModelFamily]:
    """Class decorator that registers a ModelFamily subclass by its family_name."""
    _FAMILIES[cls.family_name] = cls
    return cls


def get_family_cls(name: str) -> type[ModelFamily]:
    """Look up a registered family class by name."""
    if name not in _FAMILIES:
        raise KeyError(
            f"Unknown family '{name}'. Registered: {list(_FAMILIES)}"
        )
    return _FAMILIES[name]


def registered_families() -> dict[str, type[ModelFamily]]:
    """Return a copy of the registry."""
    return dict(_FAMILIES)
