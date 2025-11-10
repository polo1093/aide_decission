"""Utilitaires communs pour la gestion des états."""
from __future__ import annotations

from typing import Any, Mapping, Optional


def extract_scan_value(scan_table: Mapping[str, Any], key: str) -> Optional[str]:
    raw = scan_table.get(key)
    if isinstance(raw, Mapping):
        return raw.get("value")
    return raw


__all__ = ["extract_scan_value"]
