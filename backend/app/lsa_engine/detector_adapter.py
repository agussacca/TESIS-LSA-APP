from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.features import build_empty_frame_features, build_frame_features
from core.hand_detector import HandPoseDetector


@dataclass(frozen=True)
class DetectionProcessResult:
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

    features: np.ndarray
    feature_vector_size: int
    features_ready: bool

    raw_result: Any


class DetectorAdapter:
    """
    Adaptador entre el backend web y el detector original del prototipo.

    Responsabilidades:
    - inicializar HandPoseDetector una sola vez;
    - ejecutar detección sobre frames BGR;
    - construir features por frame usando core/features.py;
    - devolver una estructura estable para FrameProcessor.
    """

    def __init__(self):
        self.detector = HandPoseDetector()
        self._last_timestamp_ms = -1

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_ms: int,
        expected_hands: int = 2,
    ) -> DetectionProcessResult:
        timestamp_ms = self._ensure_monotonic_timestamp(timestamp_ms)

        detection_result = self.detector.detect_for_video(
            frame_bgr=frame_bgr,
            timestamp_ms=timestamp_ms,
        )

        try:
            features, info = build_frame_features(
                detection_result,
                expected_hands=expected_hands,
            )
        except Exception:
            features, info = build_empty_frame_features(
                expected_hands=expected_hands,
            )

        return DetectionProcessResult(
            detected_hands=int(info.detected_hands),
            used_hands=int(info.used_hands),
            expected_hands=int(info.expected_hands),

            pose_detected=bool(info.pose_detected),
            face_present=bool(info.face_present),
            body_present=bool(info.body_present),

            handedness_labels=list(info.handedness_labels),
            handedness_scores=[float(v) for v in info.handedness_scores],
            primary_label=info.primary_label,
            secondary_label=info.secondary_label,

            features=features,
            feature_vector_size=int(len(features)),
            features_ready=bool(features is not None and len(features) > 0),

            raw_result=detection_result,
        )

    def _ensure_monotonic_timestamp(self, timestamp_ms: int) -> int:
        """
        MediaPipe en modo VIDEO exige timestamps crecientes.
        Si por alguna razón llega un timestamp repetido o menor, se corrige.
        """

        timestamp_ms = int(timestamp_ms)

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def close(self) -> None:
        self.detector.close()