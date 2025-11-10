"""Gestion de l'état des captures et paramètres OCR."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from objet.entities.card import CardObservation


@dataclass
class CaptureState:
    """Paramètres liés aux captures et aux zones OCR."""

    table_capture: Dict[str, Any] = field(default_factory=dict)
    regions: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)
    templates: Dict[str, Any] = field(default_factory=dict)
    reference_path: Optional[str] = None
    card_observations: Dict[str, CardObservation] = field(default_factory=dict)
    workflow: Optional[str] = None

    def update_from_coordinates(
        self,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
    ) -> None:
        if table_capture is not None:
            self.table_capture = dict(table_capture)
        if regions is not None:
            self.regions = OrderedDict(regions)
        if templates is not None:
            self.templates = dict(templates)
        if reference_path is not None:
            self.reference_path = reference_path

    def record_observation(self, base_key: str, observation: CardObservation) -> None:
        self.card_observations[base_key] = observation

    @property
    def size(self) -> Optional[List[int]]:
        size = self.table_capture.get("size") if isinstance(self.table_capture, dict) else None
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return [int(size[0]), int(size[1])]
        return None

    @property
    def ref_offset(self) -> Optional[List[int]]:
        offset = self.table_capture.get("ref_offset") if isinstance(self.table_capture, dict) else None
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            return [int(offset[0]), int(offset[1])]
        return None


__all__ = ["CaptureState"]
