from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.lsa_engine.capture_buffer import CaptureBuffer
from app.lsa_engine.captured_item import CapturedFrameItem
from app.lsa_engine.detector_adapter import DetectorAdapter
from app.lsa_engine.model_runner import ModelRunner
from app.lsa_engine.sequence_builder import SequenceBuilder

from core.dataset_io import load_categories, get_category_map
from core.dynamic_gesture import validate_dynamic_gesture_from_captured_items
from core.static_geometry import validate_static_gesture_from_captured_items


STATE_IDLE = "IDLE"
STATE_STABILIZING = "STABILIZING"
STATE_RECORDING = "RECORDING"
STATE_WAIT_RELEASE = "WAIT_RELEASE"

MIN_FRAMES_CAPTURED = 8
MIN_CONFIDENCE_ACCEPT = 0.70
MIN_HAND_RATIO_EXPECTED = 0.90
MIN_TWO_HAND_RATIO = 0.90


@dataclass(frozen=True)
class EvaluateSessionConfig:
    target_label: str
    auto_stop_seconds: float = 2.0
    stabilize_seconds: float = 1.0
    start_consecutive_frames: int = 10
    release_consecutive_frames: int = 8
    min_frames_captured: int = MIN_FRAMES_CAPTURED


class EvaluateSession:
    """
    Sesión web de evaluación guiada.

    Replica la lógica principal de evaluate_live_abecedario.py:
    - espera cantidad de manos requerida;
    - estabiliza;
    - graba un intento durante auto_stop_seconds;
    - construye sequence shape=(20, 121);
    - ejecuta predicción raw;
    - aplica calidad, validación dinámica y validación geométrica;
    - espera retiro de manos.
    """

    def __init__(self, config: EvaluateSessionConfig):
        self.config = config
        self.target_label = config.target_label.strip().upper()

        self.detector_adapter = DetectorAdapter()
        self.sequence_builder = SequenceBuilder()
        self.model_runner = ModelRunner()
        self.capture_buffer = CaptureBuffer()

        self.label_to_hands = self._load_label_to_hands()
        self.required_hands_target = self.label_to_hands.get(self.target_label, 1)

        self.state = STATE_IDLE
        self.start_counter = 0
        self.no_hand_counter = 0

        self.state_start_timestamp_ms: int | None = None
        self.recording_start_timestamp_ms: int | None = None

        self.frame_index = 0
        self.last_result: dict | None = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_ms: int,
        orientation: str | None = None,
        mirrored: bool | None = None,
    ) -> dict:
        timestamp_ms = int(timestamp_ms)

        detection = self.detector_adapter.process_frame(
            frame_bgr,
            timestamp_ms=timestamp_ms,
            expected_hands=self.required_hands_target,
        )

        live_detected_hands = int(detection.detected_hands)

        if live_detected_hands >= self.required_hands_target:
            self.start_counter += 1
        else:
            self.start_counter = 0

        if live_detected_hands == 0:
            self.no_hand_counter += 1
        else:
            self.no_hand_counter = 0

        event = None
        final_result = None

        if self.state == STATE_IDLE:
            if self.start_counter >= self.config.start_consecutive_frames:
                self.state = STATE_STABILIZING
                self.state_start_timestamp_ms = timestamp_ms
                self.capture_buffer.reset()
                self.last_result = None
                event = "stabilizing_started"

        elif self.state == STATE_STABILIZING:
            if live_detected_hands < self.required_hands_target:
                self._reset_to_idle()
                event = "stabilizing_cancelled"

            elif self._elapsed_seconds(self.state_start_timestamp_ms, timestamp_ms) >= self.config.stabilize_seconds:
                self.state = STATE_RECORDING
                self.state_start_timestamp_ms = timestamp_ms
                self.recording_start_timestamp_ms = timestamp_ms
                self.capture_buffer.reset()
                event = "recording_started"

        elif self.state == STATE_RECORDING:
            captured_item = self._build_captured_item(
                frame_bgr=frame_bgr,
                timestamp_ms=timestamp_ms,
                detection=detection,
                orientation=orientation,
                mirrored=mirrored,
            )
            self.capture_buffer.add(captured_item)

            elapsed_recording = self._elapsed_seconds(
                self.recording_start_timestamp_ms,
                timestamp_ms,
            )

            if elapsed_recording >= self.config.auto_stop_seconds:
                final_result = self._finalize_attempt()
                self.last_result = final_result

                self.state = STATE_WAIT_RELEASE
                self.state_start_timestamp_ms = timestamp_ms
                self.recording_start_timestamp_ms = None
                self.start_counter = 0
                self.no_hand_counter = 0
                self.capture_buffer.reset()
                event = "attempt_finalized"

        elif self.state == STATE_WAIT_RELEASE:
            if self.no_hand_counter >= self.config.release_consecutive_frames:
                self._reset_to_idle()
                event = "released_ready_for_next_attempt"

        self.frame_index += 1

        response = {
            "type": "evaluate_update",
            "target_label": self.target_label,
            "state": self.state,
            "event": event,

            "required_hands_target": self.required_hands_target,
            "live_detected_hands": live_detected_hands,
            "start_counter": self.start_counter,
            "start_consecutive_frames": self.config.start_consecutive_frames,
            "no_hand_counter": self.no_hand_counter,
            "release_consecutive_frames": self.config.release_consecutive_frames,

            "captured_items_count": self.capture_buffer.count,
            "model_loaded": self.model_runner.model_loaded,
            "model_error": self.model_runner.load_error,

            "hands_detected": detection.detected_hands,
            "used_hands": detection.used_hands,
            "pose_detected": detection.pose_detected,
            "face_present": detection.face_present,
            "body_present": detection.body_present,
            "handedness_labels": detection.handedness_labels,
            "handedness_scores": detection.handedness_scores,
            "primary_label": detection.primary_label,
            "secondary_label": detection.secondary_label,

            "attempt_finalized": final_result is not None,
            "final_result": final_result,
            "last_result": self.last_result,
        }

        return response

    def _finalize_attempt(self) -> dict:
        captured_items = self.capture_buffer.to_legacy_captured_items()
        frames_captured = len(captured_items)

        if frames_captured < self.config.min_frames_captured:
            return {
                "accepted": False,
                "correct": False,
                "quality_ok": False,
                "quality_message": (
                    f"Intento demasiado corto: "
                    f"{frames_captured} < {self.config.min_frames_captured} frames."
                ),
                "frames_captured": frames_captured,
                "pred_label": None,
                "confidence": None,
            }

        sequence_result = self.sequence_builder.build_sequence(
            captured_items,
            expected_hands_for_features=self.required_hands_target,
            min_frames_captured=self.config.min_frames_captured,
        )

        prediction = self.model_runner.predict(sequence_result.sequence)

        pred_label = prediction.pred_label
        confidence = float(prediction.confidence or 0.0)

        quality_message, quality_ok = self._classify_capture_quality(
            target_label=self.target_label,
            pred_label=pred_label,
            confidence=confidence,
            stats=sequence_result.stats,
        )

        dynamic_result = validate_dynamic_gesture_from_captured_items(
            captured_items=captured_items,
            label=self.target_label,
            expected_hands=self.required_hands_target,
        )

        if dynamic_result.required:
            quality_message += " | " + dynamic_result.message

            if not dynamic_result.ok:
                quality_ok = False

        static_result = validate_static_gesture_from_captured_items(
            captured_items=captured_items,
            label=self.target_label,
            expected_hands=self.required_hands_target,
        )

        if static_result.required:
            quality_message += " | " + static_result.message

            if not static_result.ok:
                quality_ok = False

        correct = pred_label == self.target_label
        accepted = bool(correct and quality_ok)

        return {
            "target_label": self.target_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "correct": correct,
            "quality_ok": quality_ok,
            "accepted": accepted,
            "quality_message": quality_message,

            "frames_captured": frames_captured,
            "sampled_frames": sequence_result.stats.get("sampled_frames", 0),
            "sequence_shape": tuple(sequence_result.sequence.shape),

            "expected_hands_for_features": self.required_hands_target,
            "expected_hands_estimated": sequence_result.stats.get("expected_hands_estimated", 1),
            "hand_ratio_any": sequence_result.stats.get("hand_ratio_any", 0.0),
            "hand_ratio_expected": sequence_result.stats.get("hand_ratio_expected", 0.0),
            "avg_detected_hands": sequence_result.stats.get("avg_detected_hands", 0.0),
            "two_hand_ratio": sequence_result.stats.get("two_hand_ratio", 0.0),
            "pose_ratio": sequence_result.stats.get("pose_ratio", 0.0),
            "face_anchor_ratio": sequence_result.stats.get("face_anchor_ratio", 0.0),
            "body_anchor_ratio": sequence_result.stats.get("body_anchor_ratio", 0.0),

            "dynamic": self._serialize_dynamic_result(dynamic_result),
            "static": self._serialize_static_result(static_result),

            "top_predictions": prediction.top_predictions,
        }

    def _classify_capture_quality(
        self,
        *,
        target_label: str,
        pred_label: str | None,
        confidence: float,
        stats: dict,
    ) -> tuple[str, bool]:
        issues = []

        required_hands_target = self.label_to_hands.get(target_label)
        required_hands_pred = self.label_to_hands.get(pred_label) if pred_label else None

        estimated_hands = int(stats["expected_hands_estimated"])

        if confidence < MIN_CONFIDENCE_ACCEPT:
            issues.append(
                f"confianza baja ({confidence:.2f} < {MIN_CONFIDENCE_ACCEPT:.2f})"
            )

        if required_hands_target is not None:
            if estimated_hands != required_hands_target:
                issues.append(
                    f"{target_label} requiere {required_hands_target} mano(s), "
                    f"pero se estimaron {estimated_hands}"
                )

            if required_hands_target == 1:
                if stats["hand_ratio_expected"] < MIN_HAND_RATIO_EXPECTED:
                    issues.append(
                        f"hand_ratio_expected bajo para {target_label} "
                        f"({stats['hand_ratio_expected']:.2f} < {MIN_HAND_RATIO_EXPECTED:.2f})"
                    )

            elif required_hands_target == 2:
                if stats["two_hand_ratio"] < MIN_TWO_HAND_RATIO:
                    issues.append(
                        f"{target_label} requiere dos manos visibles "
                        f"({stats['two_hand_ratio']:.2f} < {MIN_TWO_HAND_RATIO:.2f})"
                    )

                if stats["avg_detected_hands"] < 1.80:
                    issues.append(
                        f"promedio de manos detectadas bajo para {target_label} "
                        f"({stats['avg_detected_hands']:.2f} < 1.80)"
                    )

        if required_hands_pred is not None:
            if required_hands_pred == 2 and stats["two_hand_ratio"] < MIN_TWO_HAND_RATIO:
                issues.append(
                    f"la predicción {pred_label} requiere dos manos, "
                    f"pero se detectaron dos manos en pocos frames "
                    f"({stats['two_hand_ratio']:.2f} < {MIN_TWO_HAND_RATIO:.2f})"
                )

        if issues:
            return "Intento NO CONFIABLE: " + " | ".join(issues), False

        return "Intento confiable", True

    def _build_captured_item(
        self,
        *,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        detection: Any,
        orientation: str | None,
        mirrored: bool | None,
    ) -> CapturedFrameItem:
        return CapturedFrameItem(
            frame_index=self.frame_index,
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

            frame_bgr=frame_bgr.copy(),
            orientation=orientation,
            mirrored=mirrored,
        )

    def _serialize_dynamic_result(self, result: Any) -> dict:
        return {
            "required": result.required,
            "ok": result.ok,
            "message": result.message,
            "movement_total": result.movement_total,
            "x_range": result.x_range,
            "y_range": result.y_range,
            "axis_range": result.axis_range,
            "valid_hand_ratio": result.valid_hand_ratio,
            "valid_points": result.valid_points,
            "total_points": result.total_points,
            "reasons": result.reasons or [],
        }

    def _serialize_static_result(self, result: Any) -> dict:
        return {
            "required": result.required,
            "ok": result.ok,
            "message": result.message,
            "expected_hands": result.expected_hands,
            "frames_used": result.frames_used,
            "valid_frames": result.valid_frames,
            "valid_ratio": result.valid_ratio,
            "reasons": result.reasons or [],
        }

    def _elapsed_seconds(
        self,
        start_timestamp_ms: int | None,
        current_timestamp_ms: int,
    ) -> float:
        if start_timestamp_ms is None:
            return 0.0

        return max(0.0, (current_timestamp_ms - start_timestamp_ms) / 1000.0)

    def _reset_to_idle(self) -> None:
        self.state = STATE_IDLE
        self.state_start_timestamp_ms = None
        self.recording_start_timestamp_ms = None
        self.capture_buffer.reset()
        self.start_counter = 0
        self.no_hand_counter = 0

    def _load_label_to_hands(self) -> dict[str, int]:
        categories = load_categories()
        category_map = get_category_map(categories)

        return {
            label: category_map[label].hands
            for label in self.model_runner.labels
            if label in category_map
        }

    def close(self) -> None:
        self.detector_adapter.close()