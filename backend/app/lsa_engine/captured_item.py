from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CapturedFrameItem:
    """
    Representa un frame procesado dentro de un intento o segmento de reconocimiento.

    Mantiene compatibilidad conceptual con captured_items de evaluate_live_abecedario.py:
        {
            "frame": frame.copy(),
            "result": live_result,
        }
    """

    frame_index: int
    timestamp_ms: int

    result: Any
    features: np.ndarray

    detected_hands: int
    used_hands: int
    expected_hands: int

    pose_detected: bool
    face_present: bool
    body_present: bool

    handedness_labels: list[str]
    handedness_scores: list[float]
    primary_label: str | None
    secondary_label: str | None

    frame_bgr: np.ndarray | None = None
    orientation: str | None = None
    mirrored: bool | None = None

    def to_legacy_dict(self) -> dict:
        item = {
            "result": self.result,
            "features": self.features,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
        }

        if self.frame_bgr is not None:
            item["frame"] = self.frame_bgr

        return item