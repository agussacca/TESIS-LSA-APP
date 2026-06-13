from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from config import FEATURES_PER_FRAME, FRAMES_PER_VIDEO
from app.lsa_engine.capture_buffer import CaptureBuffer
from app.lsa_engine.captured_item import CapturedFrameItem
from app.lsa_engine.detector_adapter import DetectorAdapter
from app.lsa_engine.model_runner import ModelRunner
from app.lsa_engine.sequence_builder import SequenceBuilder


@dataclass(frozen=True)
class FrameMetadata:
    width: int
    height: int
    channels: int
    size_bytes: int | None = None
    mirrored: bool | None = None
    orientation: str | None = None
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class FrameProcessResult:
    frame_ready: bool
    image_width: int
    image_height: int
    image_channels: int

    mean_b: float
    mean_g: float
    mean_r: float

    hands_detected: int | None
    used_hands: int | None
    expected_hands: int | None

    pose_detected: bool | None
    face_present: bool | None
    body_present: bool | None

    handedness_labels: list[str]
    handedness_scores: list[float]
    primary_label: str | None
    secondary_label: str | None

    features_ready: bool
    feature_vector_size: int | None

    captured_items_count: int
    capture_duration_ms: int | None

    sequence_ready: bool
    sequence_shape: tuple[int, int] | None
    sampled_frames: int

    expected_hands_estimated: int
    hand_ratio_any: float
    hand_ratio_expected: float
    avg_detected_hands: float
    two_hand_ratio: float
    pose_ratio: float
    face_anchor_ratio: float
    body_anchor_ratio: float
    avg_anchor_scale: float

    model_ready: bool
    model_loaded: bool
    model_path: str | None
    model_error: str | None

    pred_label: str | None
    confidence: float | None
    top_predictions: list[dict]
    accepted: bool

    message: str
    debug: dict[str, Any]


class FrameProcessor:
    """
    Procesador de frames del motor LSA.

    Etapa actual:
    - recibe imagen OpenCV BGR;
    - ejecuta detector manos/pose;
    - construye features de 121 valores;
    - acumula captured_items completos;
    - construye secuencia GRU shape=(20, 121);
    - ejecuta predicción raw con el modelo GRU si la secuencia está lista.
    """

    def __init__(self):
        self.detector_adapter = DetectorAdapter()
        self.capture_buffer = CaptureBuffer()
        self.sequence_builder = SequenceBuilder()
        self.model_runner = ModelRunner()
        self._frame_index = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        metadata: FrameMetadata,
        expected_hands: int = 2,
        min_frames_captured: int = 8,
    ) -> FrameProcessResult:
        self._validate_frame(frame_bgr)

        height, width, channels = frame_bgr.shape
        mean_bgr = frame_bgr.mean(axis=(0, 1))

        _frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(metadata.timestamp_ms or self._frame_index)

        detection = self.detector_adapter.process_frame(
            frame_bgr,
            timestamp_ms=timestamp_ms,
            expected_hands=expected_hands,
        )

        captured_item = CapturedFrameItem(
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
            result=detection.raw_result,
            features=detection.features,

            detected_hands=detection.detected_hands,
            used_hands=detection.used_hands,
            expected_hands=detection.expected_hands,

            pose_detected=detection.pose_detected,
            face_present=detection.face_present,
            body_present=detection.body_present,

            handedness_labels=detection.handedness_labels,
            handedness_scores=detection.handedness_scores,
            primary_label=detection.primary_label,
            secondary_label=detection.secondary_label,

            orientation=metadata.orientation,
            mirrored=metadata.mirrored,
        )

        self.capture_buffer.add(captured_item)
        capture_state = self.capture_buffer.state

        sequence_result = self.sequence_builder.build_sequence(
            self.capture_buffer.to_legacy_captured_items(),
            expected_hands_for_features=expected_hands,
            min_frames_captured=min_frames_captured,
        )

        prediction = None

        if sequence_result.ready_for_model:
            prediction = self.model_runner.predict(sequence_result.sequence)

        self._frame_index += 1

        sequence_shape = (
            tuple(sequence_result.sequence.shape)
            if sequence_result.sequence is not None
            else None
        )

        model_ready = bool(
            sequence_result.ready_for_model
            and self.model_runner.model_loaded
        )

        stats = sequence_result.stats

        pred_label = prediction.pred_label if prediction is not None else None
        confidence = prediction.confidence if prediction is not None else None
        top_predictions = prediction.top_predictions if prediction is not None else []

        model_error = None
        if prediction is not None and prediction.error:
            model_error = prediction.error
        elif self.model_runner.load_error:
            model_error = self.model_runner.load_error

        return FrameProcessResult(
            frame_ready=True,
            image_width=int(width),
            image_height=int(height),
            image_channels=int(channels),

            mean_b=round(float(mean_bgr[0]), 2),
            mean_g=round(float(mean_bgr[1]), 2),
            mean_r=round(float(mean_bgr[2]), 2),

            hands_detected=detection.detected_hands,
            used_hands=detection.used_hands,
            expected_hands=detection.expected_hands,

            pose_detected=detection.pose_detected,
            face_present=detection.face_present,
            body_present=detection.body_present,

            handedness_labels=detection.handedness_labels,
            handedness_scores=detection.handedness_scores,
            primary_label=detection.primary_label,
            secondary_label=detection.secondary_label,

            features_ready=detection.features_ready,
            feature_vector_size=detection.feature_vector_size,

            captured_items_count=capture_state.captured_items_count,
            capture_duration_ms=capture_state.duration_ms,

            sequence_ready=sequence_result.ready_for_model,
            sequence_shape=sequence_shape,
            sampled_frames=int(stats.get("sampled_frames", 0)),

            expected_hands_estimated=int(stats.get("expected_hands_estimated", expected_hands)),
            hand_ratio_any=float(stats.get("hand_ratio_any", 0.0)),
            hand_ratio_expected=float(stats.get("hand_ratio_expected", 0.0)),
            avg_detected_hands=float(stats.get("avg_detected_hands", 0.0)),
            two_hand_ratio=float(stats.get("two_hand_ratio", 0.0)),
            pose_ratio=float(stats.get("pose_ratio", 0.0)),
            face_anchor_ratio=float(stats.get("face_anchor_ratio", 0.0)),
            body_anchor_ratio=float(stats.get("body_anchor_ratio", 0.0)),
            avg_anchor_scale=float(stats.get("avg_anchor_scale", 0.0)),

            model_ready=model_ready,
            model_loaded=self.model_runner.model_loaded,
            model_path=str(self.model_runner.model_path),
            model_error=model_error,

            pred_label=pred_label,
            confidence=confidence,
            top_predictions=top_predictions,
            accepted=False,

            message=(
                "Frame procesado correctamente: secuencia GRU preparada "
                "y predicción raw ejecutada si hay suficientes frames."
            ),
            debug={
                "metadata_width": metadata.width,
                "metadata_height": metadata.height,
                "metadata_channels": metadata.channels,
                "metadata_size_bytes": metadata.size_bytes,
                "metadata_mirrored": metadata.mirrored,
                "metadata_orientation": metadata.orientation,
                "timestamp_ms": metadata.timestamp_ms,
                "rgb_conversion_ready": True,
                "feature_vector_size": detection.feature_vector_size,
                "features_dtype": str(detection.features.dtype),
                "frames_per_video": FRAMES_PER_VIDEO,
                "features_per_frame": FEATURES_PER_FRAME,
                "sequence_shape": sequence_shape,
                "sequence_dtype": str(sequence_result.sequence.dtype),
                "ready_for_model_min_frames": min_frames_captured,
                "model_loaded": self.model_runner.model_loaded,
                "model_path": str(self.model_runner.model_path),
                "model_error": model_error,
            },
        )

    def reset_capture(self) -> None:
        self.capture_buffer.reset()
        self._frame_index = 0

    def _validate_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            raise ValueError("El frame recibido es None.")

        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("El frame debe ser un np.ndarray.")

        if frame_bgr.ndim != 3:
            raise ValueError(
                f"El frame debe tener 3 dimensiones. Shape recibido: {frame_bgr.shape}"
            )

        if frame_bgr.shape[2] != 3:
            raise ValueError(
                f"El frame debe tener 3 canales BGR. Shape recibido: {frame_bgr.shape}"
            )

        if frame_bgr.size == 0:
            raise ValueError("El frame está vacío.")

    def close(self) -> None:
        self.detector_adapter.close()