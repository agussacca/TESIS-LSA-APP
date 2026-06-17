#sequence_builder.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import FRAMES_PER_VIDEO, FEATURES_PER_FRAME
from core.features import build_empty_frame_features, build_frame_features


@dataclass(frozen=True)
class SequenceBuildResult:
    sequence: np.ndarray
    stats: dict
    ready_for_model: bool


def get_sampled_indices_from_buffer(total_frames: int) -> list[int]:
    """
    Replica la lógica de evaluate_live_abecedario.py.

    Si hay más frames que FRAMES_PER_VIDEO, se toman 20 índices distribuidos
    uniformemente entre el inicio y el final del intento.
    """

    if total_frames <= 0:
        return []

    if total_frames <= FRAMES_PER_VIDEO:
        return list(range(total_frames))

    return np.linspace(
        0,
        total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int,
    ).tolist()


def estimate_expected_hands(info_list) -> int:
    """
    Estima si el intento parece de una o dos manos.

    Si al menos la mitad de los frames muestreados tienen dos manos,
    estima 2; si no, estima 1.
    """

    if not info_list:
        return 1

    detected_counts = [info.detected_hands for info in info_list]
    frames_with_two = sum(1 for count in detected_counts if count >= 2)
    ratio_two = frames_with_two / len(detected_counts)

    return 2 if ratio_two >= 0.50 else 1


class SequenceBuilder:
    """
    Construye una secuencia fija para el modelo secuencial de reconocimiento
    a partir de todos los captured_items del intento.

    La salida es compatible con modelos recurrentes GRU o LSTM, siempre que
    hayan sido entrenados con la misma cantidad de frames, la misma cantidad
    de features por frame y el mismo procedimiento de normalización.

    Entrada:
        captured_items completos del intento/segmento.

    Salida:
        sequence shape=(FRAMES_PER_VIDEO, FEATURES_PER_FRAME), actualmente (20, 121).

    Modos:
        - Evaluate / deletreo guiado:
            expected_hands_for_features viene definido por la letra objetivo.

        - Deletreo libre:
            expected_hands_for_features=None, por lo que se estima si el intento
            parece de una o dos manos antes de construir la secuencia final.
    """

    def build_sequence(
        self,
        captured_items: list[dict],
        *,
        expected_hands_for_features: int | None = None,
        min_frames_captured: int = 8,
    ) -> SequenceBuildResult:
        sampled_indices = get_sampled_indices_from_buffer(len(captured_items))

        if not sampled_indices:
            effective_expected_hands = (
                int(expected_hands_for_features)
                if expected_hands_for_features is not None
                else 1
            )

            empty, _ = build_empty_frame_features(
                expected_hands=effective_expected_hands
            )

            sequence = np.tile(empty, (FRAMES_PER_VIDEO, 1)).astype(np.float32)

            return SequenceBuildResult(
                sequence=sequence,
                stats=self._empty_stats(
                    expected_hands_for_features=effective_expected_hands
                ),
                ready_for_model=False,
            )

        sampled_results = [
            captured_items[idx]["result"]
            for idx in sampled_indices
        ]

        initial_infos = []

        # Primera pasada:
        # Se usa expected_hands=2 para estimar si el intento parece de una
        # o dos manos, sin depender todavía de una letra objetivo.
        for result in sampled_results:
            _, info = build_frame_features(
                result=result,
                expected_hands=2,
            )
            initial_infos.append(info)

        expected_hands_estimated = estimate_expected_hands(initial_infos)

        effective_expected_hands = (
            int(expected_hands_for_features)
            if expected_hands_for_features is not None
            else int(expected_hands_estimated)
        )

        frames_with_two_hands = sum(
            1 for info in initial_infos
            if info.detected_hands >= 2
        )

        sampled_count = len(sampled_results)

        two_hand_ratio = (
            frames_with_two_hands / sampled_count
            if sampled_count > 0
            else 0.0
        )

        frame_features = []
        frames_with_any_hand = 0
        frames_with_expected_hands = 0
        total_detected_hands = 0

        frames_with_pose = 0
        frames_with_face_anchor = 0
        frames_with_body_anchor = 0
        anchor_scale_values = []

        # Segunda pasada:
        # Se construye la secuencia definitiva con la cantidad de manos efectiva:
        # - fija si viene de evaluate/guided spell;
        # - estimada si viene de free spell.
        for result in sampled_results:
            features, info = build_frame_features(
                result=result,
                expected_hands=effective_expected_hands,
            )

            frame_features.append(features)

            if info.detected_hands > 0:
                frames_with_any_hand += 1

            if info.used_hands >= effective_expected_hands:
                frames_with_expected_hands += 1

            total_detected_hands += info.detected_hands

            pose_present = bool(
                getattr(
                    info,
                    "pose_present",
                    getattr(info, "pose_detected", False),
                )
            )
            face_present = bool(getattr(info, "face_present", False))
            body_present = bool(getattr(info, "body_present", False))
            anchor_scale = float(getattr(info, "anchor_scale", 0.0) or 0.0)

            if pose_present:
                frames_with_pose += 1

            if face_present:
                frames_with_face_anchor += 1

            if body_present:
                frames_with_body_anchor += 1

            if anchor_scale > 0:
                anchor_scale_values.append(anchor_scale)

        if len(frame_features) == 0:
            empty, _ = build_empty_frame_features(
                expected_hands=effective_expected_hands
            )
            frame_features = [empty]

        while len(frame_features) < FRAMES_PER_VIDEO:
            frame_features.append(frame_features[-1])

        sequence = np.array(frame_features[:FRAMES_PER_VIDEO], dtype=np.float32)

        if sequence.shape != (FRAMES_PER_VIDEO, FEATURES_PER_FRAME):
            raise ValueError(
                f"Sequence inválida. Esperado="
                f"({FRAMES_PER_VIDEO}, {FEATURES_PER_FRAME}), "
                f"obtenido={sequence.shape}."
            )

        stats = {
            "expected_hands_for_features": effective_expected_hands,
            "expected_hands_estimated": expected_hands_estimated,
            "hand_ratio_any": frames_with_any_hand / sampled_count if sampled_count > 0 else 0.0,
            "hand_ratio_expected": frames_with_expected_hands / sampled_count if sampled_count > 0 else 0.0,
            "avg_detected_hands": total_detected_hands / sampled_count if sampled_count > 0 else 0.0,
            "two_hand_ratio": two_hand_ratio,
            "sampled_frames": sampled_count,
            "pose_ratio": frames_with_pose / sampled_count if sampled_count > 0 else 0.0,
            "face_anchor_ratio": frames_with_face_anchor / sampled_count if sampled_count > 0 else 0.0,
            "body_anchor_ratio": frames_with_body_anchor / sampled_count if sampled_count > 0 else 0.0,
            "avg_anchor_scale": float(np.mean(anchor_scale_values)) if anchor_scale_values else 0.0,
        }

        ready_for_model = len(captured_items) >= min_frames_captured

        return SequenceBuildResult(
            sequence=sequence,
            stats=stats,
            ready_for_model=ready_for_model,
        )

    def _empty_stats(self, *, expected_hands_for_features: int | None) -> dict:
        effective_expected_hands = (
            int(expected_hands_for_features)
            if expected_hands_for_features is not None
            else 1
        )

        return {
            "expected_hands_for_features": effective_expected_hands,
            "expected_hands_estimated": effective_expected_hands,
            "hand_ratio_any": 0.0,
            "hand_ratio_expected": 0.0,
            "avg_detected_hands": 0.0,
            "two_hand_ratio": 0.0,
            "sampled_frames": 0,
            "pose_ratio": 0.0,
            "face_anchor_ratio": 0.0,
            "body_anchor_ratio": 0.0,
            "avg_anchor_scale": 0.0,
        }