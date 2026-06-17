# free_spell_session.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.lsa_engine.capture_buffer import CaptureBuffer
from app.lsa_engine.captured_item import CapturedFrameItem
from app.lsa_engine.detector_adapter import DetectorAdapter
from app.lsa_engine.model_runner import ModelRunner
from app.lsa_engine.sequence_builder import SequenceBuilder

from core.dataset_io import get_category_map, load_categories
from core.dynamic_gesture import validate_dynamic_gesture_from_captured_items
from core.static_geometry import validate_static_gesture_from_captured_items


STATE_IDLE = "IDLE"
STATE_STABILIZING = "STABILIZING"
STATE_RECORDING = "RECORDING"
STATE_WAIT_RELEASE = "WAIT_RELEASE"

MIN_FRAMES_CAPTURED = 8

MIN_HAND_RATIO_EXPECTED = 0.90
MIN_CONFIDENCE_ACCEPT = 0.70
MIN_TWO_HAND_RATIO = 0.90
MIN_TOP2_MARGIN_ACCEPT = 0.15

HAND_COUNT_AMBIGUOUS_LOW = 0.20
HAND_COUNT_AMBIGUOUS_HIGH = 0.80


@dataclass(frozen=True)
class FreeSpellSessionConfig:
    record_seconds: float = 2.0
    stabilize_seconds: float = 1.0
    start_consecutive_frames: int = 10
    release_consecutive_frames: int = 8
    start_min_hands: int = 1
    min_frames_captured: int = MIN_FRAMES_CAPTURED
    min_confidence: float = MIN_CONFIDENCE_ACCEPT
    min_margin: float = MIN_TOP2_MARGIN_ACCEPT
    append_unreliable: bool = False


class FreeSpellSession:
    """
    Sesión de deletreo libre/no guiado.

    No existe target_label. La letra candidata es pred_label.
    El flujo es:
    - esperar mano(s);
    - estabilizar;
    - capturar intento;
    - predecir letra;
    - validar calidad, dinámica y geometría contra pred_label;
    - si accepted=true, agregar pred_label al texto formado;
    - esperar retiro de manos antes de aceptar otra letra.
    """

    def __init__(self, config: FreeSpellSessionConfig | None = None):
        self.config = config or FreeSpellSessionConfig()

        self.detector_adapter = DetectorAdapter()
        self.sequence_builder = SequenceBuilder()
        self.model_runner = ModelRunner()
        self.capture_buffer = CaptureBuffer()

        self.label_to_hands: dict[str, int] = self._load_label_to_hands()

        self.state = STATE_IDLE
        self.state_start_timestamp_ms: int | None = None
        self.recording_start_timestamp_ms: int | None = None

        self.hand_present_counter = 0
        self.no_hand_counter = 0
        self.frame_index = 0

        self.spelled_text = ""
        self.last_result: dict[str, Any] | None = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_ms: int,
        orientation: str | None = None,
        mirrored: bool | None = None,
    ) -> dict[str, Any]:
        timestamp_ms = int(timestamp_ms)

        detection = self.detector_adapter.process_frame(
            frame_bgr,
            timestamp_ms=timestamp_ms,
            expected_hands=2,
        )

        live_detected_hands = int(detection.detected_hands)

        hand_condition = live_detected_hands >= self.config.start_min_hands
        release_condition = live_detected_hands == 0

        if hand_condition:
            self.hand_present_counter += 1
        else:
            self.hand_present_counter = 0

        if release_condition:
            self.no_hand_counter += 1
        else:
            self.no_hand_counter = 0

        event = None
        final_result: dict[str, Any] | None = None

        if self.state == STATE_IDLE:
            if self.hand_present_counter >= self.config.start_consecutive_frames:
                self.state = STATE_STABILIZING
                self.state_start_timestamp_ms = timestamp_ms
                self.capture_buffer.reset()
                event = "stabilizing_started"

        elif self.state == STATE_STABILIZING:
            if not hand_condition:
                self._reset_to_idle()
                event = "stabilizing_cancelled"

            elif self._elapsed_seconds(
                self.state_start_timestamp_ms,
                timestamp_ms,
            ) >= self.config.stabilize_seconds:
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

            if elapsed_recording >= self.config.record_seconds:
                final_result = self._finalize_attempt()
                self.last_result = final_result

                if final_result and final_result.get("accepted"):
                    final_pred_label = final_result.get("pred_label")

                    if isinstance(final_pred_label, str) and final_pred_label:
                        self.spelled_text += final_pred_label

                self.state = STATE_WAIT_RELEASE
                self.state_start_timestamp_ms = timestamp_ms
                self.recording_start_timestamp_ms = None
                self.capture_buffer.reset()
                self.hand_present_counter = 0
                self.no_hand_counter = 0
                event = "attempt_finalized"

        elif self.state == STATE_WAIT_RELEASE:
            if self.no_hand_counter >= self.config.release_consecutive_frames:
                self._reset_to_idle()
                event = "released_ready_for_next_letter"

        self.frame_index += 1

        return {
            "type": "free_spell_update",
            "state": self.state,
            "event": event,

            "spelled_text": self.spelled_text,

            "live_detected_hands": live_detected_hands,
            "start_min_hands": self.config.start_min_hands,
            "hand_present_counter": self.hand_present_counter,
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

    def _finalize_attempt(self) -> dict[str, Any]:
        captured_items = self.capture_buffer.to_legacy_captured_items()
        frames_captured = len(captured_items)

        if frames_captured < self.config.min_frames_captured:
            return {
                "accepted": False,
                "appended": False,
                "quality_ok": False,
                "quality_message": (
                    f"Intento demasiado corto: "
                    f"{frames_captured} < {self.config.min_frames_captured} frames."
                ),
                "pred_label": None,
                "confidence": None,
                "frames_captured": frames_captured,
                "sampled_frames": 0,
                "sequence_shape": None,
                "top_predictions": [],
            }

        sequence_result = self.sequence_builder.build_sequence(
            captured_items,
            expected_hands_for_features=None,
            min_frames_captured=self.config.min_frames_captured,
        )

        prediction = self.model_runner.predict(sequence_result.sequence)

        pred_label = prediction.pred_label
        confidence = float(prediction.confidence or 0.0)
        top2_margin = float(prediction.top2_margin or 0.0)

        if pred_label is None:
            return {
                "accepted": False,
                "appended": False,
                "quality_ok": False,
                "quality_message": (
                    prediction.error
                    or "No se obtuvo una predicción válida del modelo."
                ),
                "base_quality_message": (
                    prediction.error
                    or "No se obtuvo una predicción válida del modelo."
                ),
                "base_quality_ok": False,

                "pred_label": None,
                "confidence": prediction.confidence,

                "top2_label": prediction.top2_label,
                "top2_confidence": prediction.top2_confidence,
                "top2_margin": prediction.top2_margin,

                "frames_captured": frames_captured,
                "sampled_frames": sequence_result.stats.get("sampled_frames", 0),
                "sequence_shape": tuple(sequence_result.sequence.shape),

                "expected_hands_for_features": sequence_result.stats.get(
                    "expected_hands_for_features"
                ),
                "expected_hands_estimated": sequence_result.stats.get(
                    "expected_hands_estimated", 1
                ),
                "expected_hands_for_validation": None,

                "hand_ratio_any": sequence_result.stats.get("hand_ratio_any", 0.0),
                "hand_ratio_expected": sequence_result.stats.get(
                    "hand_ratio_expected", 0.0
                ),
                "avg_detected_hands": sequence_result.stats.get(
                    "avg_detected_hands", 0.0
                ),
                "two_hand_ratio": sequence_result.stats.get("two_hand_ratio", 0.0),
                "pose_ratio": sequence_result.stats.get("pose_ratio", 0.0),
                "face_anchor_ratio": sequence_result.stats.get(
                    "face_anchor_ratio", 0.0
                ),
                "body_anchor_ratio": sequence_result.stats.get(
                    "body_anchor_ratio", 0.0
                ),

                "dynamic": None,
                "static": None,

                "top_predictions": prediction.top_predictions,
                "model_loaded": prediction.model_loaded,
                "model_path": prediction.model_path,
                "model_error": prediction.error,
            }

        candidate_label: str = pred_label

        base_quality_message, base_quality_ok = self._classify_capture_quality(
            pred_label=candidate_label,
            confidence=confidence,
            top2_margin=top2_margin,
            stats=sequence_result.stats,
        )

        expected_hands_for_validation = int(
            self.label_to_hands.get(
                candidate_label,
                int(sequence_result.stats.get("expected_hands_estimated", 1)),
            )
        )

        dynamic_result = validate_dynamic_gesture_from_captured_items(
            captured_items=captured_items,
            label=candidate_label,
            expected_hands=expected_hands_for_validation,
        )

        quality_message = base_quality_message
        quality_ok = base_quality_ok

        if dynamic_result.required:
            quality_message += " | " + dynamic_result.message
            if not dynamic_result.ok:
                quality_ok = False

        static_result = validate_static_gesture_from_captured_items(
            captured_items=captured_items,
            label=candidate_label,
            expected_hands=expected_hands_for_validation,
        )

        if static_result.required:
            quality_message += " | " + static_result.message
            if not static_result.ok:
                quality_ok = False

        accepted = self._should_accept_result(
            quality_ok=quality_ok,
            dynamic_result=dynamic_result,
            static_result=static_result,
        )

        return {
            "pred_label": candidate_label,
            "confidence": confidence,

            "top2_label": prediction.top2_label,
            "top2_confidence": prediction.top2_confidence,
            "top2_margin": prediction.top2_margin,

            "quality_ok": quality_ok,
            "accepted": accepted,
            "appended": accepted,
            "quality_message": quality_message,
            "base_quality_message": base_quality_message,
            "base_quality_ok": base_quality_ok,

            "frames_captured": frames_captured,
            "sampled_frames": sequence_result.stats.get("sampled_frames", 0),
            "sequence_shape": tuple(sequence_result.sequence.shape),

            "expected_hands_for_features": sequence_result.stats.get(
                "expected_hands_for_features"
            ),
            "expected_hands_estimated": sequence_result.stats.get(
                "expected_hands_estimated", 1
            ),
            "expected_hands_for_validation": expected_hands_for_validation,

            "hand_ratio_any": sequence_result.stats.get("hand_ratio_any", 0.0),
            "hand_ratio_expected": sequence_result.stats.get(
                "hand_ratio_expected", 0.0
            ),
            "avg_detected_hands": sequence_result.stats.get(
                "avg_detected_hands", 0.0
            ),
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
        pred_label: str,
        confidence: float,
        top2_margin: float,
        stats: dict[str, Any],
    ) -> tuple[str, bool]:
        issues: list[str] = []

        required_hands = self.label_to_hands.get(pred_label)

        if confidence < self.config.min_confidence:
            issues.append(
                f"confianza baja ({confidence:.2f} < {self.config.min_confidence:.2f})"
            )

        if top2_margin < self.config.min_margin:
            issues.append(
                f"margen top1-top2 bajo ({top2_margin:.2f} < {self.config.min_margin:.2f})"
            )

        if stats["hand_ratio_expected"] < MIN_HAND_RATIO_EXPECTED:
            issues.append(
                f"hand_ratio_expected bajo "
                f"({stats['hand_ratio_expected']:.2f} < {MIN_HAND_RATIO_EXPECTED:.2f})"
            )

        two_hand_ratio = float(stats["two_hand_ratio"])

        if HAND_COUNT_AMBIGUOUS_LOW <= two_hand_ratio <= HAND_COUNT_AMBIGUOUS_HIGH:
            issues.append(
                f"cantidad de manos inestable durante la captura "
                f"(two_hand_ratio={two_hand_ratio:.2f})"
            )

        if required_hands is not None:
            estimated_hands = int(stats["expected_hands_estimated"])

            if estimated_hands != required_hands:
                issues.append(
                    f"{pred_label} requiere {required_hands} mano(s), "
                    f"pero se estimaron {estimated_hands}"
                )

            if required_hands == 2:
                if stats["two_hand_ratio"] < MIN_TWO_HAND_RATIO:
                    issues.append(
                        f"pocos frames con dos manos "
                        f"({stats['two_hand_ratio']:.2f} < {MIN_TWO_HAND_RATIO:.2f})"
                    )

                if stats["avg_detected_hands"] < 1.80:
                    issues.append(
                        f"promedio de manos detectadas bajo "
                        f"({stats['avg_detected_hands']:.2f} < 1.80)"
                    )

        if issues:
            return "Intento NO CONFIABLE: " + " | ".join(issues), False

        return "Intento confiable", True

    def _should_accept_result(
        self,
        *,
        quality_ok: bool,
        dynamic_result: Any,
        static_result: Any,
    ) -> bool:
        if getattr(dynamic_result, "required", False) and not getattr(
            dynamic_result, "ok", True
        ):
            return False

        if getattr(static_result, "required", False) and not getattr(
            static_result, "ok", True
        ):
            return False

        return bool(quality_ok or self.config.append_unreliable)

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

    def delete_last_character(self) -> None:
        if self.spelled_text:
            self.spelled_text = self.spelled_text[:-1]

    def append_space(self) -> None:
        self.spelled_text += " "

    def clear_text(self) -> None:
        self.spelled_text = ""

    def reset_state_only(self) -> None:
        self._reset_to_idle()
        self.last_result = None

    def _serialize_dynamic_result(self, result: Any) -> dict[str, Any]:
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

    def _serialize_static_result(self, result: Any) -> dict[str, Any]:
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
        self.hand_present_counter = 0
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