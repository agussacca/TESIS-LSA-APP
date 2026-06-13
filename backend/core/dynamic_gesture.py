#dynamic_gesture.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.features import (
    _split_detection_result,
    _collect_detected_hands,
    _select_hands_for_class,
)


# ============================================================
# Validación dinámica para letras con movimiento - Abecedario LSA
# ============================================================
#
# Objetivo:
#   Evitar aceptar como válidas letras dinámicas cuando solo se realiza
#   una pose estática inicial/intermedia/final.
#
# Letras dinámicas actuales:
#   H, J, Z
#
# Estrategia:
#   - El modelo GRU sigue prediciendo normalmente.
#   - Para letras dinámicas, se calcula movimiento real de la mano primaria.
#   - Si el movimiento no supera umbrales mínimos, el intento se marca
#     como no confiable.
#
# Modos compatibles:
#
#   1) validate_dynamic_prediction(...)
#      Usa una secuencia ya construida para el modelo:
#        shape = (FRAMES_PER_VIDEO, FEATURES_PER_FRAME)
#      Es compatible con scripts anteriores.
#
#   2) validate_dynamic_gesture_from_captured_items(...)
#      Usa todos los frames capturados del intento.
#      Esta es la opción recomendada para evaluate_live_abecedario.py,
#      predict_auto_abecedario.py y spell_auto_abecedario.py cuando
#      se están usando detecciones live.
#
# Nota importante sobre H:
#   La validación de H ahora tiene una regla específica de trayectoria.
#   Ya no alcanza con que exista movimiento total y rango de movimiento.
#   Se validan también ambos ejes activos, dirección global compatible,
#   eficiencia de trayectoria y proporción de movimiento en X.
#
# Nota importante sobre J:
#   La validación de J ahora tiene una regla específica de trayectoria.
#   La J real se modela como un arco suave bajo la mandíbula:
#   desplazamiento lateral dominante, curvatura moderada, dirección global
#   compatible y trayectoria no recta/diagonal/vertical.
#
# Nota importante sobre Z:
#   La validación de Z ahora tiene una regla específica de trayectoria.
#   Ya no alcanza con que exista movimiento total y rango de movimiento.
#   Se validan también dirección global, ambos ejes activos, eficiencia
#   de trayectoria y cambios de dirección.
# ============================================================


DYNAMIC_LABELS = {"H", "J", "Z"}


# Umbrales iniciales calibrados con:
#   - debug_dynamic_abecedario.py
#   - debug_dynamic_trajectory_abecedario.py
#
# ignore_initial_frames:
#   Descarta algunos frames iniciales para que el ingreso a la pose
#   no cuente como movimiento dinámico válido.
#
# smooth_window:
#   Suaviza la trayectoria para reducir jitter del detector.
#
# direction_eps:
#   Umbral mínimo para considerar que un delta de X/Y indica dirección real.
#   Sirve para no contar micro-jitter como cambios de dirección.
#
# H:
#   Tiene reglas adicionales de trayectoria.
#   Estos umbrales se basan en la corrida:
#     H full / prepare_move / wrong_vertical / wrong_diagonal /
#     wrong_horizontal / wrong_inverted / wrong_partial / wrong_free.
#
# J:
#   Tiene reglas adicionales de trayectoria.
#   Estos umbrales se basan en la corrida:
#     J full / prepare_move / wrong_vertical / wrong_diagonal /
#     wrong_horizontal / wrong_inverted / wrong_partial / wrong_free.
#
# Z:
#   Tiene reglas adicionales de trayectoria.
#   Estos umbrales se basan en la corrida limpia de Z:
#     full / prepare_move / wrong_vertical / wrong_diagonal /
#     wrong_horizontal / wrong_inverted / wrong_partial / wrong_free.
DYNAMIC_RULES = {
    "H": {
        # Reglas básicas de magnitud.
        "min_movement_total": 0.12,
        "min_axis_range": 0.070,
        "min_valid_hand_ratio": 0.70,

        # Preprocesamiento.
        #
        # Para H dejamos 0 porque la corrida de debug ya se hizo
        # con pre_roll_frames=0 y stabilize_seconds=1.0.
        # Además, en H el movimiento completo puede arrancar apenas
        # comienza la captura.
        "ignore_initial_frames": 0,
        "smooth_window": 5,
        "direction_eps": 0.003,

        # Reglas específicas de trayectoria H.
        #
        # La H real, según tus pruebas, activa ambos ejes.
        "min_x_range": 0.060,
        "min_y_range": 0.060,

        # Dirección global observada para tu H real:
        #   net_dx positivo
        #   net_dy positivo
        #
        # Esto ayuda a rechazar preparación, H invertida,
        # verticales/horizontales aisladas y movimientos hacia abajo.
        "min_net_dx": 0.060,
        "min_net_dy": 0.030,

        # La H real no fue una línea casi recta en las pruebas.
        # Esto ayuda a rechazar diagonales directas y movimientos
        # de preparación muy lineales.
        "max_path_efficiency": 0.85,

        # Balance aproximado del movimiento.
        # La H no debería ser casi todo vertical ni casi todo horizontal.
        "min_x_motion_fraction": 0.22,
        "max_x_motion_fraction": 0.62,
    },
    "J": {
        # Reglas básicas de magnitud.
        "min_movement_total": 0.12,
        "min_axis_range": 0.085,
        "min_valid_hand_ratio": 0.70,

        # Preprocesamiento.
        #
        # Para J dejamos 0 porque la corrida de debug se hizo con
        # pre_roll_frames=0 y stabilize_seconds=1.0.
        # Además, la J completa puede arrancar apenas comienza la captura.
        "ignore_initial_frames": 0,
        "smooth_window": 5,
        "direction_eps": 0.003,

        # Reglas específicas de trayectoria J.
        #
        # La J real observada sigue la mandíbula con un arco suave:
        # desplazamiento lateral dominante, pero con curvatura moderada.
        "min_x_range": 0.085,
        "min_y_range": 0.0365, #0.040,
        "max_y_range": 0.120,

        # Dirección global observada para tu J real:
        #   net_dx negativo
        #   net_dy cercano a cero, con leve tolerancia hacia arriba/abajo.
        #
        # Esto ayuda a rechazar J invertida, preparación vertical,
        # diagonales directas y movimientos hacia zonas no compatibles.
        "max_net_dx": -0.070,
        "min_net_dy": -0.055,
        "max_net_dy": 0.040,

        # La J real no es una línea totalmente recta ni un movimiento
        # demasiado errático.
        "min_path_efficiency": 0.55,
        "max_path_efficiency": 0.80,

        # La curvatura bajo la mandíbula aparece como al menos un cambio
        # suave de dirección en Y. En X no debería haber zigzag.
        "min_dir_changes_y": 1,
        "max_dir_changes_x": 4,

        # Balance aproximado del movimiento.
        # La J no debería ser casi todo vertical ni una horizontal rígida.
        "min_x_motion_fraction": 0.45,
        "max_x_motion_fraction": 0.65,
    },
    "Z": {
        # Reglas básicas de magnitud.
        "min_movement_total": 0.24, #0.26,
        "min_axis_range": 0.075, #0.085,
        "min_valid_hand_ratio": 0.70,

        # Preprocesamiento.
        "ignore_initial_frames": 5,
        "smooth_window": 5,
        "direction_eps": 0.003,

        # Reglas específicas de trayectoria Z.
        #
        # La Z real, en tus pruebas, activa ambos ejes.
        "min_x_range": 0.075, #0.085,
        "min_y_range": 0.070, #0.080,

        # Dirección global observada para tu Z:
        #   net_dx negativo
        #   net_dy positivo
        #
        # Esto ayuda a rechazar Z invertida, movimiento parcial y
        # movimientos de preparación.
        "max_net_dx": -0.065, #-0.075,
        "min_net_dy": 0.060, #0.070,

        # La Z no debe ser una línea casi recta.
        # prepare_move / líneas simples suelen tener eficiencia alta.
        #
        # Tampoco debe ser demasiado errática.
        # wrong_free o movimientos muy libres pueden tener eficiencia
        # excesivamente baja.
        "min_path_efficiency": 0.30, #0.34,
        "max_path_efficiency": 0.65,

        # La Z necesita cambios de dirección en X.
        # No se exige dir_changes_y porque en algunas Z reales muestreadas
        # puede quedar en 0 según suavizado/muestreo.
        "min_dir_changes_x": 2,

        # Balance aproximado del movimiento.
        # No debe ser casi todo vertical ni casi todo horizontal.
        # Se mantiene relativamente amplio para no rechazar Z reales.
        "min_x_motion_fraction": 0.40,
        "max_x_motion_fraction": 0.85,
    },
}


# Índices V1 dentro del vector de features.
PRIMARY_HAND_PRESENT_IDX = 0
PRIMARY_WRIST_X_IDX = 1
PRIMARY_WRIST_Y_IDX = 2

# Índices V2:
# V1 = 93
# Pose anchors = 10
# Primary hand-anchor relation empieza en 103:
#   103 = hand_present
#   104 = hand_center_x
#   105 = hand_center_y
V2_PRIMARY_HAND_ANCHOR_OFFSET = 93 + 10
V2_PRIMARY_HAND_PRESENT_IDX = V2_PRIMARY_HAND_ANCHOR_OFFSET
V2_PRIMARY_HAND_CENTER_X_IDX = V2_PRIMARY_HAND_ANCHOR_OFFSET + 1
V2_PRIMARY_HAND_CENTER_Y_IDX = V2_PRIMARY_HAND_ANCHOR_OFFSET + 2


@dataclass
class DynamicValidationResult:
    label: str
    required: bool
    ok: bool
    message: str

    movement_total: float = 0.0
    x_range: float = 0.0
    y_range: float = 0.0
    axis_range: float = 0.0
    valid_hand_ratio: float = 0.0
    valid_points: int = 0
    total_points: int = 0

    min_movement_total: float = 0.0
    min_axis_range: float = 0.0
    min_valid_hand_ratio: float = 0.0

    # Métricas de trayectoria.
    net_dx: float = 0.0
    net_dy: float = 0.0
    net_distance: float = 0.0
    path_efficiency: float = 0.0
    dir_changes_x: int = 0
    dir_changes_y: int = 0
    x_motion_total: float = 0.0
    y_motion_total: float = 0.0
    x_motion_fraction: float = 0.0
    y_motion_fraction: float = 0.0
    dominant_axis: str = ""

    # Umbrales específicos opcionales.
    min_x_range: float = 0.0
    min_y_range: float = 0.0
    max_y_range: float = 0.0
    min_net_dx: float = 0.0
    max_net_dx: float = 0.0
    min_net_dy: float = 0.0
    max_net_dy: float = 0.0
    min_path_efficiency: float = 0.0
    max_path_efficiency: float = 0.0
    min_dir_changes_x: int = 0
    min_dir_changes_y: int = 0
    max_dir_changes_x: int = 0
    min_x_motion_fraction: float = 0.0
    max_x_motion_fraction: float = 0.0

    reasons: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "required": self.required,
            "ok": self.ok,
            "message": self.message,

            "movement_total": self.movement_total,
            "x_range": self.x_range,
            "y_range": self.y_range,
            "axis_range": self.axis_range,
            "valid_hand_ratio": self.valid_hand_ratio,
            "valid_points": self.valid_points,
            "total_points": self.total_points,

            "min_movement_total": self.min_movement_total,
            "min_axis_range": self.min_axis_range,
            "min_valid_hand_ratio": self.min_valid_hand_ratio,

            "net_dx": self.net_dx,
            "net_dy": self.net_dy,
            "net_distance": self.net_distance,
            "path_efficiency": self.path_efficiency,
            "dir_changes_x": self.dir_changes_x,
            "dir_changes_y": self.dir_changes_y,
            "x_motion_total": self.x_motion_total,
            "y_motion_total": self.y_motion_total,
            "x_motion_fraction": self.x_motion_fraction,
            "y_motion_fraction": self.y_motion_fraction,
            "dominant_axis": self.dominant_axis,

            "min_x_range": self.min_x_range,
            "min_y_range": self.min_y_range,
            "max_y_range": self.max_y_range,
            "min_net_dx": self.min_net_dx,
            "max_net_dx": self.max_net_dx,
            "min_net_dy": self.min_net_dy,
            "max_net_dy": self.max_net_dy,
            "min_path_efficiency": self.min_path_efficiency,
            "max_path_efficiency": self.max_path_efficiency,
            "min_dir_changes_x": self.min_dir_changes_x,
            "min_dir_changes_y": self.min_dir_changes_y,
            "max_dir_changes_x": self.max_dir_changes_x,
            "min_x_motion_fraction": self.min_x_motion_fraction,
            "max_x_motion_fraction": self.max_x_motion_fraction,

            "reasons": self.reasons or [],
        }

    def get(self, key: str, default=None):
        """
        Permite usar DynamicValidationResult como si fuera un dict.

        Ejemplo:
            dynamic_result.get("required", False)

        Esto mantiene compatibilidad con código nuevo sin romper código viejo
        que usa dynamic_result.required.
        """

        return self.to_dict().get(key, default)


def is_dynamic_label(label: str) -> bool:
    """
    Indica si una letra requiere validación dinámica.
    """

    return str(label).upper() in DYNAMIC_LABELS


def _extract_primary_hand_track(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae trayectoria 2D de la mano primaria desde una secuencia de features.

    Retorna:
        points: array shape (T, 2)
        valid_mask: array bool shape (T,)

    En V2 se prioriza hand_center_x/y.
    En V1 se usa wrist_x/y.
    """

    seq = np.asarray(sequence, dtype=np.float32)

    if seq.ndim != 2:
        raise ValueError(
            f"La secuencia debe tener shape (frames, features). "
            f"Shape recibido: {seq.shape}"
        )

    total_features = seq.shape[1]

    # V2: usar centro de mano primaria si existen esos índices.
    if total_features > V2_PRIMARY_HAND_CENTER_Y_IDX:
        presence = seq[:, V2_PRIMARY_HAND_PRESENT_IDX]
        x = seq[:, V2_PRIMARY_HAND_CENTER_X_IDX]
        y = seq[:, V2_PRIMARY_HAND_CENTER_Y_IDX]

        valid_mask = (
            (presence > 0.5)
            & np.isfinite(x)
            & np.isfinite(y)
        )

        points = np.stack([x, y], axis=1)
        return points, valid_mask

    # V1 fallback: usar muñeca primaria.
    presence = seq[:, PRIMARY_HAND_PRESENT_IDX]
    x = seq[:, PRIMARY_WRIST_X_IDX]
    y = seq[:, PRIMARY_WRIST_Y_IDX]

    valid_mask = (
        (presence > 0.5)
        & np.isfinite(x)
        & np.isfinite(y)
    )

    points = np.stack([x, y], axis=1)
    return points, valid_mask


def _smooth_points(points: np.ndarray, window: int) -> np.ndarray:
    """
    Suavizado por media móvil simple.

    Reduce jitter pequeño de MediaPipe que puede inflar artificialmente
    movement_total.
    """

    points = np.asarray(points, dtype=np.float32)

    if window <= 1:
        return points

    if len(points) < window:
        return points

    smoothed = []
    half = window // 2

    for idx in range(len(points)):
        start = max(0, idx - half)
        end = min(len(points), idx + half + 1)

        chunk = points[start:end]
        smoothed.append(np.mean(chunk, axis=0))

    return np.asarray(smoothed, dtype=np.float32)


def _count_direction_changes(values: np.ndarray, eps: float) -> int:
    """
    Cuenta cambios de dirección en una serie de deltas.

    Ignora deltas pequeños para no contar jitter como cambio real.
    """

    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return 0

    signs = np.zeros_like(values, dtype=np.int32)
    signs[values > eps] = 1
    signs[values < -eps] = -1

    non_zero_signs = signs[signs != 0]

    if len(non_zero_signs) < 2:
        return 0

    changes = int(np.sum(non_zero_signs[1:] != non_zero_signs[:-1]))

    return changes


def _empty_metrics() -> dict:
    return {
        "movement_total": 0.0,
        "x_range": 0.0,
        "y_range": 0.0,
        "axis_range": 0.0,
        "valid_hand_ratio": 0.0,
        "valid_points": 0,
        "total_points": 0,

        "net_dx": 0.0,
        "net_dy": 0.0,
        "net_distance": 0.0,
        "path_efficiency": 0.0,
        "dir_changes_x": 0,
        "dir_changes_y": 0,
        "x_motion_total": 0.0,
        "y_motion_total": 0.0,
        "x_motion_fraction": 0.0,
        "y_motion_fraction": 0.0,
        "dominant_axis": "",
    }


def _compute_metrics_from_points(
    points: np.ndarray,
    valid_mask: np.ndarray,
    *,
    smooth_window: int = 1,
    direction_eps: float = 0.003,
) -> dict:
    """
    Calcula métricas de movimiento a partir de puntos y máscara de validez.
    """

    points = np.asarray(points, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    total_points = len(points)

    if total_points == 0:
        return _empty_metrics()

    valid_points = points[valid_mask]
    valid_count = len(valid_points)

    valid_hand_ratio = (
        valid_count / total_points
        if total_points > 0
        else 0.0
    )

    if valid_count < 2:
        metrics = _empty_metrics()
        metrics.update(
            {
                "valid_hand_ratio": float(valid_hand_ratio),
                "valid_points": int(valid_count),
                "total_points": int(total_points),
            }
        )
        return metrics

    valid_points = _smooth_points(valid_points, smooth_window)

    if len(valid_points) < 2:
        metrics = _empty_metrics()
        metrics.update(
            {
                "valid_hand_ratio": float(valid_hand_ratio),
                "valid_points": int(valid_count),
                "total_points": int(total_points),
            }
        )
        return metrics

    deltas = np.diff(valid_points, axis=0)

    dx_values = deltas[:, 0]
    dy_values = deltas[:, 1]

    step_distances = np.linalg.norm(deltas, axis=1)
    movement_total = float(np.sum(step_distances))

    x_values = valid_points[:, 0]
    y_values = valid_points[:, 1]

    x_range = float(np.max(x_values) - np.min(x_values))
    y_range = float(np.max(y_values) - np.min(y_values))
    axis_range = float(max(x_range, y_range))

    net_dx = float(valid_points[-1, 0] - valid_points[0, 0])
    net_dy = float(valid_points[-1, 1] - valid_points[0, 1])
    net_distance = float(np.linalg.norm(valid_points[-1] - valid_points[0]))

    if movement_total > 0:
        path_efficiency = float(net_distance / movement_total)
    else:
        path_efficiency = 0.0

    dir_changes_x = _count_direction_changes(dx_values, direction_eps)
    dir_changes_y = _count_direction_changes(dy_values, direction_eps)

    x_motion_total = float(np.sum(np.abs(dx_values)))
    y_motion_total = float(np.sum(np.abs(dy_values)))

    total_axis_motion = x_motion_total + y_motion_total

    if total_axis_motion > 0:
        x_motion_fraction = float(x_motion_total / total_axis_motion)
        y_motion_fraction = float(y_motion_total / total_axis_motion)
    else:
        x_motion_fraction = 0.0
        y_motion_fraction = 0.0

    dominant_axis = "x" if x_range >= y_range else "y"

    return {
        "movement_total": movement_total,
        "x_range": x_range,
        "y_range": y_range,
        "axis_range": axis_range,
        "valid_hand_ratio": float(valid_hand_ratio),
        "valid_points": int(valid_count),
        "total_points": int(total_points),

        "net_dx": net_dx,
        "net_dy": net_dy,
        "net_distance": net_distance,
        "path_efficiency": path_efficiency,
        "dir_changes_x": int(dir_changes_x),
        "dir_changes_y": int(dir_changes_y),
        "x_motion_total": x_motion_total,
        "y_motion_total": y_motion_total,
        "x_motion_fraction": x_motion_fraction,
        "y_motion_fraction": y_motion_fraction,
        "dominant_axis": dominant_axis,
    }


def compute_movement_metrics(sequence: np.ndarray) -> dict:
    """
    Calcula métricas simples de movimiento usando la secuencia del modelo.

    Esta función queda por compatibilidad con scripts anteriores.
    """

    points, valid_mask = _extract_primary_hand_track(sequence)

    return _compute_metrics_from_points(
        points=points,
        valid_mask=valid_mask,
        smooth_window=1,
        direction_eps=0.003,
    )


def _get_primary_hand_center_from_result(
    result: Any,
    expected_hands: int,
) -> tuple[float, float] | None:
    """
    Extrae el centro de la mano primaria desde un resultado de MediaPipe.

    Usa la misma lógica de selección que features.py:
    - separa hand_result / pose_result;
    - colecta manos detectadas;
    - selecciona primaria/secundaria según expected_hands.
    """

    hand_result, _pose_result = _split_detection_result(result)

    hands = _collect_detected_hands(hand_result)

    if not hands:
        return None

    primary, _secondary = _select_hands_for_class(
        hands=hands,
        expected_hands=expected_hands,
    )

    if primary is None:
        return None

    return (
        float(primary["center_x"]),
        float(primary["center_y"]),
    )


def _build_dynamic_result(
    *,
    label: str,
    required: bool,
    ok: bool,
    metrics: dict | None = None,
    rule: dict | None = None,
    reasons: list[str] | None = None,
    message: str | None = None,
) -> DynamicValidationResult:
    """
    Construye un DynamicValidationResult normalizado.
    """

    metrics = metrics or {}
    rule = rule or {}
    reasons = reasons or []

    normalized_label = str(label).upper()

    if message is None:
        if not required:
            message = "Validación dinámica no requerida."
        elif ok:
            if normalized_label in {"H", "J", "Z"}:
                message = (
                    f"Validación dinámica OK para {normalized_label}: "
                    f"movement_total={metrics.get('movement_total', 0.0):.4f}, "
                    f"x_range={metrics.get('x_range', 0.0):.4f}, "
                    f"y_range={metrics.get('y_range', 0.0):.4f}, "
                    f"net_dx={metrics.get('net_dx', 0.0):.4f}, "
                    f"net_dy={metrics.get('net_dy', 0.0):.4f}, "
                    f"path_efficiency={metrics.get('path_efficiency', 0.0):.4f}, "
                    f"dir_changes_x={int(metrics.get('dir_changes_x', 0))}, "
                    f"dir_changes_y={int(metrics.get('dir_changes_y', 0))}, "
                    f"x_motion_fraction={metrics.get('x_motion_fraction', 0.0):.3f}, "
                    f"valid_hand_ratio={metrics.get('valid_hand_ratio', 0.0):.3f}"
                )
            else:
                message = (
                    f"Validación dinámica OK para {normalized_label}: "
                    f"movement_total={metrics.get('movement_total', 0.0):.4f}, "
                    f"axis_range={metrics.get('axis_range', 0.0):.4f}, "
                    f"valid_hand_ratio={metrics.get('valid_hand_ratio', 0.0):.3f}"
                )
        else:
            message = (
                f"Validación dinámica NO OK para {normalized_label}: "
                + " | ".join(reasons)
            )

    return DynamicValidationResult(
        label=normalized_label,
        required=required,
        ok=ok,
        message=message,

        movement_total=float(metrics.get("movement_total", 0.0)),
        x_range=float(metrics.get("x_range", 0.0)),
        y_range=float(metrics.get("y_range", 0.0)),
        axis_range=float(metrics.get("axis_range", 0.0)),
        valid_hand_ratio=float(metrics.get("valid_hand_ratio", 0.0)),
        valid_points=int(metrics.get("valid_points", 0)),
        total_points=int(metrics.get("total_points", 0)),

        min_movement_total=float(rule.get("min_movement_total", 0.0)),
        min_axis_range=float(rule.get("min_axis_range", 0.0)),
        min_valid_hand_ratio=float(rule.get("min_valid_hand_ratio", 0.0)),

        net_dx=float(metrics.get("net_dx", 0.0)),
        net_dy=float(metrics.get("net_dy", 0.0)),
        net_distance=float(metrics.get("net_distance", 0.0)),
        path_efficiency=float(metrics.get("path_efficiency", 0.0)),
        dir_changes_x=int(metrics.get("dir_changes_x", 0)),
        dir_changes_y=int(metrics.get("dir_changes_y", 0)),
        x_motion_total=float(metrics.get("x_motion_total", 0.0)),
        y_motion_total=float(metrics.get("y_motion_total", 0.0)),
        x_motion_fraction=float(metrics.get("x_motion_fraction", 0.0)),
        y_motion_fraction=float(metrics.get("y_motion_fraction", 0.0)),
        dominant_axis=str(metrics.get("dominant_axis", "")),

        min_x_range=float(rule.get("min_x_range", 0.0)),
        min_y_range=float(rule.get("min_y_range", 0.0)),
        max_y_range=float(rule.get("max_y_range", 0.0)),
        min_net_dx=float(rule.get("min_net_dx", 0.0)),
        max_net_dx=float(rule.get("max_net_dx", 0.0)),
        min_net_dy=float(rule.get("min_net_dy", 0.0)),
        max_net_dy=float(rule.get("max_net_dy", 0.0)),
        min_path_efficiency=float(rule.get("min_path_efficiency", 0.0)),
        max_path_efficiency=float(rule.get("max_path_efficiency", 0.0)),
        min_dir_changes_x=int(rule.get("min_dir_changes_x", 0)),
        min_dir_changes_y=int(rule.get("min_dir_changes_y", 0)),
        max_dir_changes_x=int(rule.get("max_dir_changes_x", 0)),
        min_x_motion_fraction=float(rule.get("min_x_motion_fraction", 0.0)),
        max_x_motion_fraction=float(rule.get("max_x_motion_fraction", 0.0)),

        reasons=reasons,
    )


def _append_basic_dynamic_reasons(
    *,
    label: str,
    metrics: dict,
    rule: dict,
    reasons: list[str],
) -> None:
    """
    Agrega razones de rechazo básicas por magnitud/cantidad de mano visible.
    """

    _ = label

    min_movement_total = float(rule["min_movement_total"])
    min_axis_range = float(rule["min_axis_range"])
    min_valid_hand_ratio = float(rule["min_valid_hand_ratio"])

    if metrics["valid_hand_ratio"] < min_valid_hand_ratio:
        reasons.append(
            f"mano visible insuficiente "
            f"({metrics['valid_hand_ratio']:.3f} < {min_valid_hand_ratio:.3f})"
        )

    if metrics["movement_total"] < min_movement_total:
        reasons.append(
            f"movimiento total insuficiente "
            f"({metrics['movement_total']:.4f} < {min_movement_total:.4f})"
        )

    if metrics["axis_range"] < min_axis_range:
        reasons.append(
            f"rango de movimiento insuficiente "
            f"({metrics['axis_range']:.4f} < {min_axis_range:.4f})"
        )


def _append_h_trajectory_reasons(
    *,
    metrics: dict,
    rule: dict,
    reasons: list[str],
) -> None:
    """
    Agrega razones específicas para validar trayectoria de H.

    La regla está pensada para la H observada en las pruebas:
    - movimiento con ambos ejes activos;
    - dirección global hacia X positivo e Y positivo;
    - trayectoria no demasiado recta;
    - movimiento no puramente vertical ni puramente horizontal.

    Importante:
    Esta validación NO verifica todavía la configuración de dedos.
    Solo valida trayectoria global de la mano primaria.
    """

    min_x_range = float(rule.get("min_x_range", 0.0))
    min_y_range = float(rule.get("min_y_range", 0.0))
    min_net_dx = float(rule.get("min_net_dx", 0.0))
    min_net_dy = float(rule.get("min_net_dy", 0.0))
    max_path_efficiency = float(rule.get("max_path_efficiency", 0.0))
    min_x_motion_fraction = float(rule.get("min_x_motion_fraction", 0.0))
    max_x_motion_fraction = float(rule.get("max_x_motion_fraction", 1.0))

    x_range = float(metrics.get("x_range", 0.0))
    y_range = float(metrics.get("y_range", 0.0))
    net_dx = float(metrics.get("net_dx", 0.0))
    net_dy = float(metrics.get("net_dy", 0.0))
    path_efficiency = float(metrics.get("path_efficiency", 0.0))
    x_motion_fraction = float(metrics.get("x_motion_fraction", 0.0))

    if x_range < min_x_range:
        reasons.append(
            f"H requiere movimiento suficiente en X "
            f"({x_range:.4f} < {min_x_range:.4f})"
        )

    if y_range < min_y_range:
        reasons.append(
            f"H requiere movimiento suficiente en Y "
            f"({y_range:.4f} < {min_y_range:.4f})"
        )

    if net_dx < min_net_dx:
        reasons.append(
            f"dirección global X incompatible con H "
            f"({net_dx:.4f} < {min_net_dx:.4f})"
        )

    if net_dy < min_net_dy:
        reasons.append(
            f"dirección global Y incompatible con H "
            f"({net_dy:.4f} < {min_net_dy:.4f})"
        )

    if max_path_efficiency > 0 and path_efficiency > max_path_efficiency:
        reasons.append(
            f"trayectoria demasiado directa para H "
            f"({path_efficiency:.4f} > {max_path_efficiency:.4f})"
        )

    if x_motion_fraction < min_x_motion_fraction:
        reasons.append(
            f"proporción de movimiento en X baja para H "
            f"({x_motion_fraction:.3f} < {min_x_motion_fraction:.3f})"
        )

    if x_motion_fraction > max_x_motion_fraction:
        reasons.append(
            f"proporción de movimiento en X alta para H "
            f"({x_motion_fraction:.3f} > {max_x_motion_fraction:.3f})"
        )


def _append_j_trajectory_reasons(
    *,
    metrics: dict,
    rule: dict,
    reasons: list[str],
) -> None:
    """
    Agrega razones específicas para validar trayectoria de J.

    La regla está pensada para la J observada en las pruebas:
    - arco suave bajo la mandíbula;
    - desplazamiento lateral dominante en X;
    - curvatura moderada en Y;
    - dirección global hacia X negativo;
    - trayectoria no demasiado recta, diagonal, vertical ni errática.

    Importante:
    Esta validación NO verifica todavía la configuración de dedos.
    Solo valida trayectoria global de la mano primaria.
    """

    min_x_range = float(rule.get("min_x_range", 0.0))
    min_y_range = float(rule.get("min_y_range", 0.0))
    max_y_range = float(rule.get("max_y_range", 0.0))
    max_net_dx = float(rule.get("max_net_dx", 0.0))
    min_net_dy = float(rule.get("min_net_dy", 0.0))
    max_net_dy = float(rule.get("max_net_dy", 0.0))
    min_path_efficiency = float(rule.get("min_path_efficiency", 0.0))
    max_path_efficiency = float(rule.get("max_path_efficiency", 0.0))
    min_dir_changes_y = int(rule.get("min_dir_changes_y", 0))
    max_dir_changes_x = int(rule.get("max_dir_changes_x", 0))
    min_x_motion_fraction = float(rule.get("min_x_motion_fraction", 0.0))
    max_x_motion_fraction = float(rule.get("max_x_motion_fraction", 1.0))

    x_range = float(metrics.get("x_range", 0.0))
    y_range = float(metrics.get("y_range", 0.0))
    net_dx = float(metrics.get("net_dx", 0.0))
    net_dy = float(metrics.get("net_dy", 0.0))
    path_efficiency = float(metrics.get("path_efficiency", 0.0))
    dir_changes_x = int(metrics.get("dir_changes_x", 0))
    dir_changes_y = int(metrics.get("dir_changes_y", 0))
    x_motion_fraction = float(metrics.get("x_motion_fraction", 0.0))

    if x_range < min_x_range:
        reasons.append(
            f"J requiere movimiento suficiente en X "
            f"({x_range:.4f} < {min_x_range:.4f})"
        )

    if y_range < min_y_range:
        reasons.append(
            f"J requiere curvatura/movimiento suficiente en Y "
            f"({y_range:.4f} < {min_y_range:.4f})"
        )

    if max_y_range > 0 and y_range > max_y_range:
        reasons.append(
            f"J tiene movimiento vertical excesivo "
            f"({y_range:.4f} > {max_y_range:.4f})"
        )

    if net_dx > max_net_dx:
        reasons.append(
            f"dirección global X incompatible con J "
            f"({net_dx:.4f} > {max_net_dx:.4f})"
        )

    if "min_net_dy" in rule and net_dy < min_net_dy:
        reasons.append(
            f"dirección global Y demasiado baja para J "
            f"({net_dy:.4f} < {min_net_dy:.4f})"
        )

    if "max_net_dy" in rule and net_dy > max_net_dy:
        reasons.append(
            f"dirección global Y demasiado alta para J "
            f"({net_dy:.4f} > {max_net_dy:.4f})"
        )

    if min_path_efficiency > 0 and path_efficiency < min_path_efficiency:
        reasons.append(
            f"trayectoria demasiado irregular para J "
            f"({path_efficiency:.4f} < {min_path_efficiency:.4f})"
        )

    if max_path_efficiency > 0 and path_efficiency > max_path_efficiency:
        reasons.append(
            f"trayectoria demasiado directa para J "
            f"({path_efficiency:.4f} > {max_path_efficiency:.4f})"
        )

    if dir_changes_y < min_dir_changes_y:
        reasons.append(
            f"curvatura en Y insuficiente para J "
            f"({dir_changes_y} < {min_dir_changes_y})"
        )

    if "max_dir_changes_x" in rule and dir_changes_x > max_dir_changes_x:
        reasons.append(
            f"cambios de dirección en X excesivos para J "
            f"({dir_changes_x} > {max_dir_changes_x})"
        )

    if x_motion_fraction < min_x_motion_fraction:
        reasons.append(
            f"proporción de movimiento en X baja para J "
            f"({x_motion_fraction:.3f} < {min_x_motion_fraction:.3f})"
        )

    if x_motion_fraction > max_x_motion_fraction:
        reasons.append(
            f"proporción de movimiento en X alta para J "
            f"({x_motion_fraction:.3f} > {max_x_motion_fraction:.3f})"
        )


def _append_z_trajectory_reasons(
    *,
    metrics: dict,
    rule: dict,
    reasons: list[str],
) -> None:
    """
    Agrega razones específicas para validar trayectoria de Z.

    La regla está pensada para la forma de Z observada en las pruebas:
    - movimiento con ambos ejes activos;
    - dirección global hacia X negativo e Y positivo;
    - trayectoria no recta, pero tampoco excesivamente errática;
    - cambios de dirección claros en X.
    """

    min_x_range = float(rule.get("min_x_range", 0.0))
    min_y_range = float(rule.get("min_y_range", 0.0))
    max_net_dx = float(rule.get("max_net_dx", 0.0))
    min_net_dy = float(rule.get("min_net_dy", 0.0))
    min_path_efficiency = float(rule.get("min_path_efficiency", 0.0))
    max_path_efficiency = float(rule.get("max_path_efficiency", 0.0))
    min_dir_changes_x = int(rule.get("min_dir_changes_x", 0))
    min_x_motion_fraction = float(rule.get("min_x_motion_fraction", 0.0))
    max_x_motion_fraction = float(rule.get("max_x_motion_fraction", 1.0))

    x_range = float(metrics.get("x_range", 0.0))
    y_range = float(metrics.get("y_range", 0.0))
    net_dx = float(metrics.get("net_dx", 0.0))
    net_dy = float(metrics.get("net_dy", 0.0))
    path_efficiency = float(metrics.get("path_efficiency", 0.0))
    dir_changes_x = int(metrics.get("dir_changes_x", 0))
    x_motion_fraction = float(metrics.get("x_motion_fraction", 0.0))

    if x_range < min_x_range:
        reasons.append(
            f"Z requiere movimiento suficiente en X "
            f"({x_range:.4f} < {min_x_range:.4f})"
        )

    if y_range < min_y_range:
        reasons.append(
            f"Z requiere movimiento suficiente en Y "
            f"({y_range:.4f} < {min_y_range:.4f})"
        )

    if net_dx > max_net_dx:
        reasons.append(
            f"dirección global X incompatible con Z "
            f"({net_dx:.4f} > {max_net_dx:.4f})"
        )

    if net_dy < min_net_dy:
        reasons.append(
            f"dirección global Y incompatible con Z "
            f"({net_dy:.4f} < {min_net_dy:.4f})"
        )

    if path_efficiency < min_path_efficiency:
        reasons.append(
            f"trayectoria demasiado irregular para Z "
            f"({path_efficiency:.4f} < {min_path_efficiency:.4f})"
        )

    if path_efficiency > max_path_efficiency:
        reasons.append(
            f"trayectoria demasiado directa para Z "
            f"({path_efficiency:.4f} > {max_path_efficiency:.4f})"
        )

    if dir_changes_x < min_dir_changes_x:
        reasons.append(
            f"cambios de dirección en X insuficientes para Z "
            f"({dir_changes_x} < {min_dir_changes_x})"
        )

    if x_motion_fraction < min_x_motion_fraction:
        reasons.append(
            f"proporción de movimiento en X baja para Z "
            f"({x_motion_fraction:.3f} < {min_x_motion_fraction:.3f})"
        )

    if x_motion_fraction > max_x_motion_fraction:
        reasons.append(
            f"proporción de movimiento en X alta para Z "
            f"({x_motion_fraction:.3f} > {max_x_motion_fraction:.3f})"
        )


def _validate_metrics_for_label(
    *,
    label: str,
    metrics: dict,
    rule: dict,
) -> DynamicValidationResult:
    """
    Aplica umbrales de una letra dinámica sobre métricas ya calculadas.
    """

    normalized_label = str(label).upper()

    reasons = []

    _append_basic_dynamic_reasons(
        label=normalized_label,
        metrics=metrics,
        rule=rule,
        reasons=reasons,
    )

    if normalized_label == "H":
        _append_h_trajectory_reasons(
            metrics=metrics,
            rule=rule,
            reasons=reasons,
        )

    if normalized_label == "J":
        _append_j_trajectory_reasons(
            metrics=metrics,
            rule=rule,
            reasons=reasons,
        )

    if normalized_label == "Z":
        _append_z_trajectory_reasons(
            metrics=metrics,
            rule=rule,
            reasons=reasons,
        )

    ok = len(reasons) == 0

    return _build_dynamic_result(
        label=normalized_label,
        required=True,
        ok=ok,
        metrics=metrics,
        rule=rule,
        reasons=reasons,
    )


def validate_dynamic_prediction(
    pred_label: str,
    sequence: np.ndarray,
) -> DynamicValidationResult:
    """
    Valida una predicción dinámica usando la secuencia del modelo.

    Esta función se mantiene por compatibilidad con scripts anteriores.

    Importante:
    - Usa solo los frames de la secuencia del modelo.
    - Si FRAMES_PER_VIDEO = 20, mide movimiento sobre 20 puntos.
    - Para evaluate_live_abecedario.py conviene usar
      validate_dynamic_gesture_from_captured_items(...), que usa todos los frames.
    """

    label = str(pred_label).upper()

    if label not in DYNAMIC_LABELS:
        return _build_dynamic_result(
            label=label,
            required=False,
            ok=True,
        )

    rule = DYNAMIC_RULES[label]

    direction_eps = float(rule.get("direction_eps", 0.003))

    points, valid_mask = _extract_primary_hand_track(sequence)

    metrics = _compute_metrics_from_points(
        points=points,
        valid_mask=valid_mask,
        smooth_window=1,
        direction_eps=direction_eps,
    )

    return _validate_metrics_for_label(
        label=label,
        metrics=metrics,
        rule=rule,
    )


def validate_dynamic_gesture_from_captured_items(
    *,
    captured_items: list[dict],
    label: str,
    expected_hands: int,
) -> DynamicValidationResult:
    """
    Valida una letra dinámica usando todos los frames capturados del intento.

    Esta función es la recomendada para scripts live.
    """

    normalized_label = str(label).upper()

    if normalized_label not in DYNAMIC_LABELS:
        return _build_dynamic_result(
            label=normalized_label,
            required=False,
            ok=True,
        )

    rule = DYNAMIC_RULES[normalized_label]

    ignore_initial_frames = int(rule.get("ignore_initial_frames", 0))
    smooth_window = int(rule.get("smooth_window", 1))
    direction_eps = float(rule.get("direction_eps", 0.003))

    usable_items = captured_items[ignore_initial_frames:]

    points = []
    valid_mask = []

    for item in usable_items:
        result = item.get("result")

        center = _get_primary_hand_center_from_result(
            result=result,
            expected_hands=expected_hands,
        )

        if center is None:
            points.append((0.0, 0.0))
            valid_mask.append(False)
        else:
            points.append(center)
            valid_mask.append(True)

    points_array = np.asarray(points, dtype=np.float32)
    valid_mask_array = np.asarray(valid_mask, dtype=bool)

    metrics = _compute_metrics_from_points(
        points=points_array,
        valid_mask=valid_mask_array,
        smooth_window=smooth_window,
        direction_eps=direction_eps,
    )

    validation_result = _validate_metrics_for_label(
        label=normalized_label,
        metrics=metrics,
        rule=rule,
    )

    return validation_result