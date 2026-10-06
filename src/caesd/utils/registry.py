"""Tiny name->factory registry so configs can select components by string."""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., object]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj  # type: ignore[assignment]
            return obj

        return deco

    def get(self, name: str) -> Callable[..., object]:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'; have {sorted(self._items)}")
        return self._items[name]
