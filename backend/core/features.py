#features.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import FEATURES_PER_FRAME


# ============================================================
# ESTRUCTURA DE FEATURES POR FRAME - V2
# ============================================================
#
# Base V1:
#
# Mano primaria:
#   - presence flag                    = 1
#   - wrist x, wrist y globales         = 2
#   - 21 landmarks relativos x,y        = 42
#   Total mano primaria                 = 45
#
# Mano secundaria:
#   - presence flag                    = 1
#   - wrist x, wrist y globales         = 2
#   - 21 landmarks relativos x,y        = 42
#   Total mano secundaria               = 45
#
# Relación entre manos:
#   - dx entre muñecas                  = 1
#   - dy entre muñecas                  = 1
#   - distancia entre muñecas           = 1
#   Total relación                      = 3
#
# Subtotal V1:
#   45 + 45 + 3 = 93
#
# Anclas corporales/faciales V2:
#   - pose_present                      = 1
#   - nose x,y                          = 2
#   - eye_center x,y                    = 2
#   - shoulder_center x,y               = 2
#   - face_width                        = 1
#   - shoulder_width                    = 1
#   - anchor_scale                      = 1
#   Total anclas                        = 10
#
# Relación mano primaria con anclas:
#   - hand_present                      = 1
#   - hand_center x,y                   = 2
#   - dx,dy,dist hand_center -> nose     = 3
#   - dx,dy,dist hand_center -> shoulder = 3
#   Total mano-anclas primaria           = 9
#
# Relación mano secundaria con anclas:
#   - mismo bloque                      = 9
#
# Total V2:
#   93 + 10 + 9 + 9 = 121
# ============================================================


HAND_VECTOR_SIZE = 45
INTER_HAND_RELATION_VECTOR_SIZE = 3
POSE_ANCHOR_VECTOR_SIZE = 10
HAND_ANCHOR_RELATION_VECTOR_SIZE = 9

POSE_LANDMARK_NOSE = 0
POSE_LANDMARK_LEFT_EYE = 2
POSE_LANDMARK_RIGHT_EYE = 5
POSE_LANDMARK_LEFT_SHOULDER = 11
POSE_LANDMARK_RIGHT_SHOULDER = 12

POSE_VISIBILITY_THRESHOLD = 0.30
MIN_ANCHOR_SCALE = 1e-6


@dataclass
class FrameFeatureInfo:
    """
    Información auxiliar sobre la extracción de features de un frame.

    Esto no entra al modelo, pero sirve para logs, auditoría y debugging.
    """

    detected_hands: int
    used_hands: int
    expected_hands: int
    handedness_labels: list[str]
    handedness_scores: list[float]
    primary_label: str | None
    secondary_label: str | None

    # Nombres principales usados por extract_abecedario_dataset.py V2.
    pose_present: bool = False
    face_present: bool = False
    body_present: bool = False
    anchor_scale: float = 0.0

    # Aliases/debug más detallados.
    pose_detected: bool = False
    nose_detected: bool = False
    eyes_detected: bool = False
    shoulders_detected: bool = False


def _split_detection_result(result: Any) -> tuple[Any, Any | None]:
    """
    Permite compatibilidad con dos tipos de entrada:

    V1:
        result = salida directa de HandDetector.detect_for_video(...)

    V2:
        result = HandPoseResult(
            hand_result=...,
            pose_result=...
        )

    Retorna:
        hand_result, pose_result
    """

    if hasattr(result, "hand_result"):
        return result.hand_result, getattr(result, "pose_result", None)

    return result, None


def _get_handedness_label_and_score(hand_result: Any, index: int) -> tuple[str | None, float]:
    """
    Obtiene etiqueta y score de handedness para una mano detectada.

    MediaPipe suele devolver etiquetas "Right" o "Left", pero no conviene
    depender ciegamente de esto porque puede fallar con cámara espejada,
    rotación de mano o poses ambiguas.
    """

    if not hand_result.handedness or index >= len(hand_result.handedness):
        return None, 0.0

    if not hand_result.handedness[index]:
        return None, 0.0

    category = hand_result.handedness[index][0]
    return category.category_name, float(category.score)


def _landmarks_to_relative_xy(hand_landmarks: Any) -> np.ndarray:
    """
    Convierte los 21 landmarks de una mano en 42 valores x,y relativos a la muñeca.

    La muñeca es landmark 0. Restar la muñeca hace que la forma de la mano sea
    más importante que la posición absoluta en la imagen.

    Luego se normaliza por el máximo valor absoluto para reducir diferencias
    de escala cuando la mano está más cerca o más lejos de la cámara.
    """

    wrist = hand_landmarks[0]

    coords = []
    for lm in hand_landmarks:
        coords.extend(
            [
                lm.x - wrist.x,
                lm.y - wrist.y,
            ]
        )

    coords = np.array(coords, dtype=np.float32)

    max_abs = float(np.max(np.abs(coords)))
    if max_abs > 0:
        coords = coords / max_abs

    return coords


def _calculate_hand_center(hand_landmarks: Any) -> tuple[float, float]:
    """
    Calcula un centro simple de la mano usando el promedio de sus 21 landmarks.

    Para relaciones espaciales con cara/cuerpo, el centro de la mano suele ser
    más estable que usar únicamente la muñeca.
    """

    xs = [float(lm.x) for lm in hand_landmarks]
    ys = [float(lm.y) for lm in hand_landmarks]

    return float(np.mean(xs)), float(np.mean(ys))


def _build_single_hand_vector(hand_landmarks: Any) -> np.ndarray:
    """
    Construye el vector de 45 features para una mano.

    Incluye:
    - flag de presencia;
    - posición global de la muñeca;
    - landmarks relativos normalizados.
    """

    wrist = hand_landmarks[0]
    relative_xy = _landmarks_to_relative_xy(hand_landmarks)

    vec = np.zeros(HAND_VECTOR_SIZE, dtype=np.float32)

    vec[0] = 1.0
    vec[1] = float(wrist.x)
    vec[2] = float(wrist.y)
    vec[3:] = relative_xy

    return vec


def _empty_hand_vector() -> np.ndarray:
    """
    Vector vacío para una mano no detectada o no utilizada.
    """

    return np.zeros(HAND_VECTOR_SIZE, dtype=np.float32)


def _collect_detected_hands(hand_result: Any) -> list[dict]:
    """
    Extrae todas las manos detectadas por MediaPipe y las guarda en una estructura
    más cómoda para el resto del pipeline.
    """

    hands = []

    if hand_result is None:
        return hands

    if not hand_result.hand_landmarks:
        return hands

    for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
        label, score = _get_handedness_label_and_score(hand_result, idx)

        wrist = hand_landmarks[0]
        center_x, center_y = _calculate_hand_center(hand_landmarks)

        hands.append(
            {
                "index": idx,
                "landmarks": hand_landmarks,
                "label": label,
                "score": score,
                "wrist_x": float(wrist.x),
                "wrist_y": float(wrist.y),
                "center_x": center_x,
                "center_y": center_y,
            }
        )

    return hands


def _select_hands_for_class(
    hands: list[dict],
    expected_hands: int,
) -> tuple[dict | None, dict | None]:
    """
    Selecciona qué manos se usarán como primaria y secundaria.

    Para letras de una mano:
    - se usa una sola mano;
    - se prioriza la detección con mayor score;
    - si MediaPipe detecta dos manos por error, la segunda se ignora.

    Para letras de dos manos:
    - se usan hasta dos manos;
    - se intenta ordenar de forma estable;
    - primero se prioriza handedness si existe Right/Left;
    - si no es confiable, se usa orden horizontal por posición x.
    """

    if not hands:
        return None, None

    if expected_hands == 1:
        primary = max(hands, key=lambda h: h["score"])
        return primary, None

    if len(hands) == 1:
        return hands[0], None

    top_hands = sorted(hands, key=lambda h: h["score"], reverse=True)[:2]

    right_hand = next((h for h in top_hands if h["label"] == "Right"), None)
    left_hand = next((h for h in top_hands if h["label"] == "Left"), None)

    if right_hand is not None and left_hand is not None:
        return right_hand, left_hand

    ordered = sorted(top_hands, key=lambda h: h["wrist_x"])
    primary = ordered[0]
    secondary = ordered[1]

    return primary, secondary


def _build_inter_hand_relation(
    primary: dict | None,
    secondary: dict | None,
) -> np.ndarray:
    """
    Construye features simples de relación entre manos.

    Si falta una de las dos manos, la relación queda en cero.
    """

    relation = np.zeros(INTER_HAND_RELATION_VECTOR_SIZE, dtype=np.float32)

    if primary is None or secondary is None:
        return relation

    dx = secondary["wrist_x"] - primary["wrist_x"]
    dy = secondary["wrist_y"] - primary["wrist_y"]
    distance = math.sqrt(dx * dx + dy * dy)

    relation[0] = float(dx)
    relation[1] = float(dy)
    relation[2] = float(distance)

    return relation


def _landmark_is_visible(lm: Any) -> bool:
    """
    Determina si un landmark de pose es utilizable.

    Algunos resultados de MediaPipe incluyen visibility/presence.
    Si esos campos no existen, se asume visible para no romper compatibilidad.
    """

    visibility = getattr(lm, "visibility", None)
    presence = getattr(lm, "presence", None)

    if visibility is not None and float(visibility) < POSE_VISIBILITY_THRESHOLD:
        return False

    if presence is not None and float(presence) < POSE_VISIBILITY_THRESHOLD:
        return False

    return True


def _get_pose_landmark(pose_landmarks: Any, index: int) -> tuple[float, float] | None:
    """
    Devuelve x,y de un landmark de pose si existe y es visible.
    """

    if pose_landmarks is None:
        return None

    if index >= len(pose_landmarks):
        return None

    lm = pose_landmarks[index]

    if not _landmark_is_visible(lm):
        return None

    return float(lm.x), float(lm.y)


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    """
    Distancia euclídea 2D entre dos puntos normalizados.
    """

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    return math.sqrt(dx * dx + dy * dy)


def _midpoint_xy(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """
    Punto medio entre dos puntos normalizados.
    """

    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _extract_pose_anchors(pose_result: Any) -> dict:
    """
    Extrae anclas corporales/faciales desde PoseLandmarker.

    Se usan:
    - nariz;
    - centro de ojos;
    - centro de hombros;
    - ancho facial aproximado;
    - ancho de hombros;
    - escala de referencia.
    """

    default = {
        "pose_present": False,
        "nose": None,
        "eye_center": None,
        "shoulder_center": None,
        "face_width": 0.0,
        "shoulder_width": 0.0,
        "anchor_scale": 0.0,
        "nose_detected": False,
        "eyes_detected": False,
        "shoulders_detected": False,
    }

    if pose_result is None:
        return default

    if not getattr(pose_result, "pose_landmarks", None):
        return default

    if len(pose_result.pose_landmarks) == 0:
        return default

    pose_landmarks = pose_result.pose_landmarks[0]

    nose = _get_pose_landmark(pose_landmarks, POSE_LANDMARK_NOSE)
    left_eye = _get_pose_landmark(pose_landmarks, POSE_LANDMARK_LEFT_EYE)
    right_eye = _get_pose_landmark(pose_landmarks, POSE_LANDMARK_RIGHT_EYE)
    left_shoulder = _get_pose_landmark(pose_landmarks, POSE_LANDMARK_LEFT_SHOULDER)
    right_shoulder = _get_pose_landmark(pose_landmarks, POSE_LANDMARK_RIGHT_SHOULDER)

    eye_center = None
    shoulder_center = None
    face_width = 0.0
    shoulder_width = 0.0

    if left_eye is not None and right_eye is not None:
        eye_center = _midpoint_xy(left_eye, right_eye)
        face_width = _distance_xy(left_eye, right_eye)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center = _midpoint_xy(left_shoulder, right_shoulder)
        shoulder_width = _distance_xy(left_shoulder, right_shoulder)

    scale_candidates = [
        value
        for value in [face_width, shoulder_width]
        if value > MIN_ANCHOR_SCALE
    ]

    if scale_candidates:
        anchor_scale = max(scale_candidates)
    else:
        anchor_scale = 1.0

    return {
        "pose_present": True,
        "nose": nose,
        "eye_center": eye_center,
        "shoulder_center": shoulder_center,
        "face_width": float(face_width),
        "shoulder_width": float(shoulder_width),
        "anchor_scale": float(anchor_scale),
        "nose_detected": nose is not None,
        "eyes_detected": eye_center is not None,
        "shoulders_detected": shoulder_center is not None,
    }


def _build_pose_anchor_vector(anchors: dict) -> np.ndarray:
    """
    Construye el vector de 10 features de anclas.
    """

    vec = np.zeros(POSE_ANCHOR_VECTOR_SIZE, dtype=np.float32)

    vec[0] = 1.0 if anchors["pose_present"] else 0.0

    nose = anchors["nose"]
    eye_center = anchors["eye_center"]
    shoulder_center = anchors["shoulder_center"]

    if nose is not None:
        vec[1] = float(nose[0])
        vec[2] = float(nose[1])

    if eye_center is not None:
        vec[3] = float(eye_center[0])
        vec[4] = float(eye_center[1])

    if shoulder_center is not None:
        vec[5] = float(shoulder_center[0])
        vec[6] = float(shoulder_center[1])

    vec[7] = float(anchors["face_width"])
    vec[8] = float(anchors["shoulder_width"])
    vec[9] = float(anchors["anchor_scale"])

    return vec


def _build_hand_anchor_relation(
    hand: dict | None,
    anchors: dict,
) -> np.ndarray:
    """
    Construye el vector de 9 features de relación mano-anclas.

    Incluye:
    - presencia de mano;
    - centro de mano global;
    - relación normalizada mano -> nariz;
    - relación normalizada mano -> centro de hombros.
    """

    vec = np.zeros(HAND_ANCHOR_RELATION_VECTOR_SIZE, dtype=np.float32)

    if hand is None:
        return vec

    hand_center = (
        float(hand["center_x"]),
        float(hand["center_y"]),
    )

    anchor_scale = float(anchors.get("anchor_scale", 1.0))
    if anchor_scale <= MIN_ANCHOR_SCALE:
        anchor_scale = 1.0

    vec[0] = 1.0
    vec[1] = hand_center[0]
    vec[2] = hand_center[1]

    nose = anchors["nose"]
    if nose is not None:
        dx = (hand_center[0] - nose[0]) / anchor_scale
        dy = (hand_center[1] - nose[1]) / anchor_scale
        dist = math.sqrt(dx * dx + dy * dy)

        vec[3] = float(dx)
        vec[4] = float(dy)
        vec[5] = float(dist)

    shoulder_center = anchors["shoulder_center"]
    if shoulder_center is not None:
        dx = (hand_center[0] - shoulder_center[0]) / anchor_scale
        dy = (hand_center[1] - shoulder_center[1]) / anchor_scale
        dist = math.sqrt(dx * dx + dy * dy)

        vec[6] = float(dx)
        vec[7] = float(dy)
        vec[8] = float(dist)

    return vec


def build_frame_features(result: Any, expected_hands: int) -> tuple[np.ndarray, FrameFeatureInfo]:
    """
    Convierte una detección de MediaPipe en un vector fijo de FEATURES_PER_FRAME.

    Parámetros:
    - result:
        V1: salida de HandDetector.detect_for_video(...).
        V2: salida de HandPoseDetector.detect_for_video(...).

    - expected_hands:
        cantidad de manos esperadas para la letra actual.
        Debe ser 1 o 2 según metadata/categories_abecedario.json.

    Retorna:
    - features: np.ndarray de shape (121,) en V2.
    - info: información auxiliar para auditoría.
    """

    if expected_hands not in (1, 2):
        raise ValueError(
            f"expected_hands debe ser 1 o 2. Valor recibido: {expected_hands}"
        )

    hand_result, pose_result = _split_detection_result(result)

    hands = _collect_detected_hands(hand_result)

    primary, secondary = _select_hands_for_class(
        hands=hands,
        expected_hands=expected_hands,
    )

    primary_vec = (
        _build_single_hand_vector(primary["landmarks"])
        if primary is not None
        else _empty_hand_vector()
    )

    secondary_vec = (
        _build_single_hand_vector(secondary["landmarks"])
        if secondary is not None
        else _empty_hand_vector()
    )

    inter_hand_relation_vec = _build_inter_hand_relation(primary, secondary)

    anchors = _extract_pose_anchors(pose_result)

    pose_anchor_vec = _build_pose_anchor_vector(anchors)

    primary_anchor_relation_vec = _build_hand_anchor_relation(
        hand=primary,
        anchors=anchors,
    )

    secondary_anchor_relation_vec = _build_hand_anchor_relation(
        hand=secondary,
        anchors=anchors,
    )

    features = np.concatenate(
        [
            primary_vec,
            secondary_vec,
            inter_hand_relation_vec,
            pose_anchor_vec,
            primary_anchor_relation_vec,
            secondary_anchor_relation_vec,
        ]
    ).astype(np.float32)

    if len(features) != FEATURES_PER_FRAME:
        raise ValueError(
            f"Vector de features inválido. "
            f"Esperado={FEATURES_PER_FRAME}, obtenido={len(features)}. "
            f"Para V2, FEATURES_PER_FRAME debe ser 121."
        )

    pose_present = bool(anchors["pose_present"])
    face_present = bool(
        anchors["nose_detected"] or anchors["eyes_detected"]
    )
    body_present = bool(anchors["shoulders_detected"])

    info = FrameFeatureInfo(
        detected_hands=len(hands),
        used_hands=int(primary is not None) + int(secondary is not None),
        expected_hands=expected_hands,
        handedness_labels=[h["label"] or "Unknown" for h in hands],
        handedness_scores=[h["score"] for h in hands],
        primary_label=primary["label"] if primary is not None else None,
        secondary_label=secondary["label"] if secondary is not None else None,

        pose_present=pose_present,
        face_present=face_present,
        body_present=body_present,
        anchor_scale=float(anchors["anchor_scale"]),

        pose_detected=pose_present,
        nose_detected=bool(anchors["nose_detected"]),
        eyes_detected=bool(anchors["eyes_detected"]),
        shoulders_detected=bool(anchors["shoulders_detected"]),
    )

    return features, info


def build_empty_frame_features(expected_hands: int) -> tuple[np.ndarray, FrameFeatureInfo]:
    """
    Devuelve un frame completamente vacío.

    Se usa cuando no se detecta ninguna mano o cuando un video no puede leerse.
    """

    features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

    info = FrameFeatureInfo(
        detected_hands=0,
        used_hands=0,
        expected_hands=expected_hands,
        handedness_labels=[],
        handedness_scores=[],
        primary_label=None,
        secondary_label=None,

        pose_present=False,
        face_present=False,
        body_present=False,
        anchor_scale=0.0,

        pose_detected=False,
        nose_detected=False,
        eyes_detected=False,
        shoulders_detected=False,
    )

    return features, info