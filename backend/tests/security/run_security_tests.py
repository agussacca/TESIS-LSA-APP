from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


SECURITY_DIR = Path(__file__).resolve().parent

GRUPOS = [
    (
        "Seguridad - Autenticación mediante JWT",
        [
            (
                "test_pa_seg_01_login_valido_genera_token",
                "PA-SEG-01 - Login válido genera token de acceso",
            ),
            (
                "test_pa_seg_02_ruta_protegida_sin_token",
                "PA-SEG-02 - Ruta protegida rechaza solicitud sin token",
            ),
            (
                "test_pa_seg_03_ruta_protegida_token_invalido",
                "PA-SEG-03 - Ruta protegida rechaza token inválido",
            ),
            (
                "test_pa_seg_04_ruta_protegida_token_valido",
                "PA-SEG-04 - Ruta protegida acepta token válido",
            ),
        ],
    ),
    (
        "Seguridad - Almacenamiento de contraseñas",
        [
            (
                "test_pa_seg_05_contrasenia_no_texto_plano",
                "PA-SEG-05 - Contraseña no almacenada en texto plano",
            ),
        ],
    ),
    (
        "Seguridad - Sesión WebSocket",
        [
            (
                "test_pa_seg_06_websocket_entrada_invalida",
                "PA-SEG-06 - Manejo controlado de entrada inválida",
            ),
        ],
    ),
]


class _RecolectorResultados:
    def __init__(self):
        self.resultados = {}

    def pytest_runtest_logreport(self, report):
        # Guardar el resultado de la llamada principal.
        if report.when == "call":
            self.resultados[report.nodeid] = (
                report.outcome,
                report.duration,
            )
            return

        # Si falla setup o teardown, conservar también ese fallo.
        if report.failed:
            self.resultados[report.nodeid] = (
                report.outcome,
                report.duration,
            )


def _buscar_resultado(resultados, nombre_funcion):
    sufijo = f"::{nombre_funcion}"
    for nodeid, resultado in resultados.items():
        if nodeid.endswith(sufijo):
            return resultado
    return None


def _imprimir_linea(etiqueta, resultado):
    if resultado is None:
        estado = "NO EJECUTADA"
        duracion_ms = 0
    else:
        outcome, duration = resultado
        if outcome == "passed":
            estado = "APROBADA"
        elif outcome == "skipped":
            estado = "OMITIDA"
        else:
            estado = "FALLIDA"

        duracion_ms = max(1, round(duration * 1000))

    prefijo = f"[{estado}] {etiqueta}"
    ancho = 78
    puntos = "." * max(4, ancho - len(prefijo))

    print(f"{prefijo} {puntos} {duracion_ms} ms")

    return estado == "APROBADA"


def main():
    recolector = _RecolectorResultados()

    salida_pytest = io.StringIO()
    error_pytest = io.StringIO()

    with redirect_stdout(salida_pytest), redirect_stderr(error_pytest):
        codigo = pytest.main(
            [
                "-q",
                str(SECURITY_DIR),
                "--disable-warnings",
                "--tb=short",
            ],
            plugins=[recolector],
        )

    todas_aprobadas = True

    for titulo, pruebas in GRUPOS:
        print()
        print(titulo)
        print("-" * 58)

        for nombre_funcion, etiqueta in pruebas:
            resultado = _buscar_resultado(
                recolector.resultados,
                nombre_funcion,
            )
            aprobada = _imprimir_linea(etiqueta, resultado)
            todas_aprobadas = todas_aprobadas and aprobada

        print("-" * 58)

    if codigo != 0 or not todas_aprobadas:
        print()
        print("Detalle de pytest")
        print("-" * 58)

        detalle = salida_pytest.getvalue().strip()
        errores = error_pytest.getvalue().strip()

        if detalle:
            print(detalle)

        if errores:
            print(errores)

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
