from __future__ import annotations

from collections import defaultdict
from typing import Any


def _get(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalizar_letra(valor: Any) -> str:
    return str(valor or "").strip().upper()


def intento_aceptado(intento: Any) -> bool:
    letra_esperada = _normalizar_letra(_get(intento, "letra_esperada"))
    letra_predicha = _normalizar_letra(_get(intento, "letra_predicha"))
    validado = bool(_get(intento, "validado", False))
    return bool(letra_esperada and letra_esperada == letra_predicha and validado)


def calcular_estadisticas_practica_por_letra(intentos: list[Any]) -> dict[str, dict[str, Any]]:
    acumulado: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_intentos": 0,
            "intentos_aceptados": 0,
        }
    )

    for intento in intentos:
        letra = _normalizar_letra(_get(intento, "letra_esperada"))
        if not letra:
            continue

        acumulado[letra]["total_intentos"] += 1
        if intento_aceptado(intento):
            acumulado[letra]["intentos_aceptados"] += 1

    resultado: dict[str, dict[str, Any]] = {}
    for letra, datos in acumulado.items():
        total = int(datos["total_intentos"])
        aceptados = int(datos["intentos_aceptados"])
        resultado[letra] = {
            "letra": letra,
            "total_intentos": total,
            "intentos_aceptados": aceptados,
            "precision": aceptados / total if total else 0.0,
        }

    return resultado


def agrupar_progreso_por_letra(intentos: list[Any]) -> list[dict[str, Any]]:
    estadisticas = calcular_estadisticas_practica_por_letra(intentos)
    return [estadisticas[letra] for letra in sorted(estadisticas.keys())]


def validar_ronda_minijuego(cantidad_minijuegos: int, correctas: int) -> None:
    if cantidad_minijuegos < 0:
        raise ValueError("La cantidad de minijuegos no puede ser negativa.")
    if correctas < 0:
        raise ValueError("La cantidad de respuestas correctas no puede ser negativa.")
    if correctas > cantidad_minijuegos:
        raise ValueError("Las respuestas correctas no pueden superar la cantidad total de minijuegos.")
    if cantidad_minijuegos == 0:
        raise ValueError("Una ronda debe incluir al menos un minijuego.")


def ronda_perfecta(ronda: Any) -> bool:
    cantidad = int(_get(ronda, "cantidad_minijuegos", 0) or 0)
    correctas = int(_get(ronda, "correctas", 0) or 0)
    return cantidad > 0 and correctas == cantidad


def _nombre_categoria(ronda: Any) -> str:
    categoria = _get(ronda, "categoria")
    if categoria is not None:
        nombre = _get(categoria, "nombre")
        if nombre:
            return str(nombre)

    nombre_directo = _get(ronda, "categoria_nombre")
    if nombre_directo:
        return str(nombre_directo)

    return str(_get(ronda, "categoria_id", "Sin categoría"))


def calcular_resumen_estadisticas(
    *,
    intentos: list[Any],
    palabras_deletreadas: list[Any],
    rondas: list[Any],
) -> dict[str, Any]:
    progreso_por_letra = agrupar_progreso_por_letra(intentos)
    senias_aprendidas = sum(1 for item in progreso_por_letra if item["intentos_aceptados"] > 0)

    rondas_por_categoria: dict[str, int] = {}
    for ronda in rondas:
        if not ronda_perfecta(ronda):
            continue
        nombre = _nombre_categoria(ronda)
        rondas_por_categoria[nombre] = rondas_por_categoria.get(nombre, 0) + 1

    return {
        "senias_aprendidas_camara": senias_aprendidas,
        "palabras_deletreadas_exitosamente": len(palabras_deletreadas),
        "rondas_por_categoria": rondas_por_categoria,
        "progreso_por_letra": progreso_por_letra,
    }
