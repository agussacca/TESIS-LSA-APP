from __future__ import annotations

from datetime import date, timedelta
from typing import Any

XP_POR_RESPUESTA_CORRECTA = 5
BONIFICACION_RONDA_PERFECTA = 10
NIVELES_XP = [0, 120, 260, 440, 660, 920, 1220, 1560, 1940, 2360]


def calcular_xp_ronda_minijuego(*, cantidad_minijuegos: int, correctas: int) -> int:
    xp = max(0, correctas) * XP_POR_RESPUESTA_CORRECTA
    if cantidad_minijuegos > 0 and correctas == cantidad_minijuegos:
        xp += BONIFICACION_RONDA_PERFECTA
    return xp


def calcular_nivel(xp_total: int) -> int:
    xp_total = max(0, int(xp_total or 0))
    nivel = 1
    for indice, umbral in enumerate(NIVELES_XP, start=1):
        if xp_total >= umbral:
            nivel = indice
    return min(nivel, 10)


def progreso_nivel(xp_total: int) -> dict[str, int]:
    nivel = calcular_nivel(xp_total)
    xp_nivel_actual = NIVELES_XP[nivel - 1]
    xp_siguiente_nivel = NIVELES_XP[nivel] if nivel < 10 else NIVELES_XP[-1]
    return {
        "nivel": nivel,
        "xp_nivel_actual": max(0, xp_total - xp_nivel_actual),
        "xp_siguiente_nivel": max(1, xp_siguiente_nivel - xp_nivel_actual),
    }


def actualizar_racha_por_objetivo_diario(
    *,
    fecha_ultima_racha: date | None,
    racha_actual: int,
    racha_maxima: int,
    fecha_actual: date,
) -> dict[str, Any]:
    if fecha_ultima_racha == fecha_actual:
        nueva_racha = racha_actual
    elif fecha_ultima_racha == fecha_actual - timedelta(days=1):
        nueva_racha = racha_actual + 1
    else:
        nueva_racha = 1

    return {
        "racha_actual": nueva_racha,
        "racha_maxima": max(racha_maxima, nueva_racha),
        "fecha_ultima_racha": fecha_actual,
    }


def objetivo_puede_otorgar_xp(
    *,
    completados: list[Any],
    objetivo_id: int,
    clave_periodo: str,
) -> bool:
    for item in completados:
        if isinstance(item, dict):
            item_objetivo_id = item.get("objetivo_id")
            item_clave_periodo = item.get("clave_periodo")
        else:
            item_objetivo_id = getattr(item, "objetivo_id", None)
            item_clave_periodo = getattr(item, "clave_periodo", None)

        if item_objetivo_id == objetivo_id and item_clave_periodo == clave_periodo:
            return False

    return True


def calcular_xp_total_desde_actividad(*, intentos_aceptados: int, palabras: int, rondas: list[Any], objetivos_xp: int = 0) -> int:
    xp = intentos_aceptados * 5
    xp += palabras * 10
    for ronda in rondas:
        cantidad = int(getattr(ronda, "cantidad_minijuegos", 0) or 0)
        correctas = int(getattr(ronda, "correctas", 0) or 0)
        xp += calcular_xp_ronda_minijuego(cantidad_minijuegos=cantidad, correctas=correctas)
    xp += objetivos_xp
    return xp
