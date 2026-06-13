from __future__ import annotations

import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


INCREMENTOS = {
    "I1": "Incremento 1 - Acceso, perfil y contenido educativo",
    "I2": "Incremento 2 - Práctica con cámara y deletreo",
    "I3": "Incremento 3 - Actividades de refuerzo y seguimiento del progreso",
    "I4": "Incremento 4 - Gamificación y personalización",
}


CASOS = {
    # ==========================================================
    # Incremento 1
    # ==========================================================
    "PU-01": {
        "tipo": "unitaria",
        "incremento": "I1",
        "nombre": "Validación de datos de usuario registrado",
        "objetivo": (
            "Verifica que los datos mínimos de un usuario registrado se validen correctamente, "
            "considerando correo electrónico, contraseña, nombre visible y foto de perfil opcional. "
            "El caso no contempla persistencia de usuario invitado, ya que el modo invitado no se "
            "registra en la base de datos."
        ),
    },
    "PI-01": {
        "tipo": "integracion",
        "incremento": "I1",
        "nombre": "Consulta de contenido educativo disponible",
        "objetivo": (
            "Verifica que la API recupere correctamente las categorías de aprendizaje y las señas "
            "asociadas desde la base de datos temporal, respetando el orden definido para su "
            "visualización."
        ),
    },

    # ==========================================================
    # Incremento 2
    # ==========================================================
    "PI-02": {
        "tipo": "integracion",
        "incremento": "I2",
        "nombre": "No persistencia de práctica realizada como invitado",
        "objetivo": (
            "Verifica que una práctica realizada sin usuario registrado no genere actividad persistida. "
            "El usuario invitado puede utilizar la práctica con cámara, pero sus intentos no se guardan "
            "ni impactan en estadísticas, progreso, objetivos o logros."
        ),
    },
    "PI-03": {
        "tipo": "integracion",
        "incremento": "I2",
        "nombre": "Registro de intento de práctica con cámara",
        "objetivo": (
            "Verifica que un intento enviado por un usuario registrado se persista conservando la letra "
            "esperada, la letra predicha y el resultado de validación del mecanismo completo, y que luego "
            "impacte en las estadísticas del usuario."
        ),
    },
    "PI-04": {
        "tipo": "integracion",
        "incremento": "I2",
        "nombre": "Registro de palabra deletreada exitosamente",
        "objetivo": (
            "Verifica que, cuando un usuario registrado completa correctamente una palabra en deletreo "
            "guiado, la palabra se registre como deletreada exitosamente para su uso posterior en "
            "estadísticas, objetivos y gamificación."
        ),
    },

    # ==========================================================
    # Incremento 3
    # ==========================================================
    "PU-03": {
        "tipo": "unitaria",
        "incremento": "I3",
        "nombre": "Resumen inicial sin actividad registrada",
        "objetivo": (
            "Verifica que el resumen de estadísticas devuelva valores iniciales en cero cuando el usuario "
            "registrado todavía no realizó prácticas, deletreos exitosos ni rondas de minijuegos."
        ),
    },
    "PU-04": {
        "tipo": "unitaria",
        "incremento": "I3",
        "nombre": "Cálculo de estadísticas de práctica por letra",
        "objetivo": (
            "Verifica el cálculo de intentos totales, intentos aceptados y precisión por letra, "
            "considerando como aceptado únicamente el intento donde la letra esperada coincide con la "
            "letra predicha y además fue validado por el mecanismo completo."
        ),
    },
    "PU-05": {
        "tipo": "unitaria",
        "incremento": "I3",
        "nombre": "Agrupación del progreso por letra",
        "objetivo": (
            "Verifica que los intentos de práctica se agrupen por letra esperada y que se calcule la "
            "precisión correspondiente a cada una, sin depender de campos derivados persistidos."
        ),
    },
    "PU-06": {
        "tipo": "unitaria",
        "incremento": "I3",
        "nombre": "Validación de ronda con cantidad inválida",
        "objetivo": (
            "Verifica que el contrato de datos rechace rondas de minijuegos con cantidades negativas "
            "o inconsistentes, por ejemplo cuando la cantidad de respuestas correctas supera la cantidad "
            "total de minijuegos."
        ),
    },
    "PI-05": {
        "tipo": "integracion",
        "incremento": "I3",
        "nombre": "Registro de intentos y consulta de progreso por letra",
        "objetivo": (
            "Verifica que los intentos registrados mediante la API se reflejen posteriormente en el "
            "progreso agrupado por letra, aplicando la regla de aceptación definida por coincidencia "
            "entre letra esperada, letra predicha y validación del mecanismo."
        ),
    },
    "PI-06": {
        "tipo": "integracion",
        "incremento": "I3",
        "nombre": "Actualización del panel de usuario con actividad registrada",
        "objetivo": (
            "Verifica que el panel del usuario refleje la actividad persistida, incluyendo señas "
            "aprendidas con cámara, palabras deletreadas exitosamente y rondas de minijuegos "
            "registradas por categoría."
        ),
    },
    "PI-07": {
        "tipo": "integracion",
        "incremento": "I3",
        "nombre": "Registro de ronda de minijuego y actualización de estadísticas",
        "objetivo": (
            "Verifica que la API registre una ronda de minijuego y que sus datos impacten en las "
            "estadísticas del usuario registrado, especialmente en el conteo de rondas completadas por "
            "categoría y rondas perfectas derivadas."
        ),
    },

    # ==========================================================
    # Incremento 4
    # ==========================================================
    "PU-07": {
        "tipo": "unitaria",
        "incremento": "I4",
        "nombre": "Cálculo de experiencia por ronda de minijuego",
        "objetivo": (
            "Verifica que el sistema calcule la experiencia obtenida en una ronda a partir de la "
            "cantidad de respuestas correctas y de la bonificación correspondiente cuando la ronda "
            "es perfecta, sin persistir la experiencia en la propia ronda."
        ),
    },
    "PU-08": {
        "tipo": "unitaria",
        "incremento": "I4",
        "nombre": "Cálculo de nivel a partir de experiencia acumulada",
        "objetivo": (
            "Verifica que la experiencia total acumulada se traduzca correctamente en uno de los diez "
            "niveles predefinidos y que se determine el progreso hacia el siguiente umbral."
        ),
    },
    "PU-09": {
        "tipo": "unitaria",
        "incremento": "I4",
        "nombre": "Actualización de racha diaria por objetivo completado",
        "objetivo": (
            "Verifica que la racha diaria se incremente cuando el usuario completa al menos un objetivo "
            "diario y que se reinicie cuando se interrumpe la continuidad definida por la fecha de última "
            "racha."
        ),
    },
    "PU-10": {
        "tipo": "unitaria",
        "incremento": "I4",
        "nombre": "Prevención de recompensa duplicada por objetivo",
        "objetivo": (
            "Verifica que un objetivo completado dentro del mismo período no otorgue experiencia más de "
            "una vez, utilizando la combinación de usuario, objetivo y clave de período."
        ),
    },
    "PI-08": {
        "tipo": "integracion",
        "incremento": "I4",
        "nombre": "Control de duplicación en sincronizaciones repetidas",
        "objetivo": (
            "Verifica que ejecutar la sincronización más de una vez no duplique experiencia, objetivos "
            "completados ni logros ya desbloqueados."
        ),
    },
    "PI-09": {
        "tipo": "integracion",
        "incremento": "I4",
        "nombre": "Cumplimiento de objetivo mediante ronda de minijuego",
        "objetivo": (
            "Verifica que una ronda de minijuego registrada pueda contribuir al cumplimiento de objetivos "
            "diarios o semanales definidos para el usuario."
        ),
    },
    "PI-10": {
        "tipo": "integracion",
        "incremento": "I4",
        "nombre": "Consulta de logros luego de sincronizar el progreso",
        "objetivo": (
            "Verifica que los logros obtenidos y pendientes puedan consultarse correctamente después de "
            "actualizar la gamificación, mostrando nombre, descripción, imagen y fecha de desbloqueo "
            "cuando corresponda."
        ),
    },
    "PI-11": {
        "tipo": "integracion",
        "incremento": "I4",
        "nombre": "Equipamiento de marco y título disponible",
        "objetivo": (
            "Verifica que el usuario registrado pueda equipar únicamente marcos y títulos disponibles "
            "según su nivel o sus rondas perfectas en Deportes, y que la selección quede reflejada en "
            "su perfil."
        ),
    },
}


def _estado(case: ET.Element) -> str:
    if case.find("failure") is not None:
        return "NO APROBADA"
    if case.find("error") is not None:
        return "ERROR"
    if case.find("skipped") is not None:
        return "OMITIDA"
    return "APROBADA"


def _codigo_original(nombre: str) -> str:
    match = re.search(r"test_(pu|pi)_(\d{2})_", nombre, flags=re.IGNORECASE)
    if not match:
        return "SIN-ID"
    return f"{match.group(1).upper()}-{match.group(2)}"


def _tipo_desde_archivo(case: ET.Element) -> str:
    file_attr = case.attrib.get("file", "")
    file_attr = file_attr.replace("\\", "/").lower()

    if "/unit/" in file_attr:
        return "unitaria"
    if "/integration/" in file_attr:
        return "integracion"
    return "otra"


def _tiempo_ms(case: ET.Element) -> int:
    try:
        return int(round(float(case.attrib.get("time", "0") or 0) * 1000))
    except ValueError:
        return 0


def _normalizar_nombre_test(nombre: str) -> str:
    return nombre.replace("test_", "").replace("_", " ").strip().capitalize()


def _numero_codigo(codigo: str) -> int:
    try:
        return int(codigo.split("-")[1])
    except Exception:
        return 999


def _orden_salida(caso: dict) -> tuple[int, int]:
    tipo_orden = 1 if caso["tipo"] == "unitaria" else 2 if caso["tipo"] == "integracion" else 9
    return tipo_orden, _numero_codigo(caso["codigo_salida"])


def _linea_estado(codigo: str, nombre: str, estado: str, ms: int) -> str:
    izquierda = f"    [{estado}] {codigo} - {nombre}"
    derecha = f"{ms} ms"
    ancho = 112
    puntos = "." * max(3, ancho - len(izquierda) - len(derecha) - 1)
    return f"{izquierda} {puntos} {derecha}"


def _leer_casos(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    testcases = list(root.iter("testcase"))

    casos = []

    for case in testcases:
        nombre_original = case.attrib.get("name", "sin_nombre")
        codigo_original = _codigo_original(nombre_original)
        metadata = CASOS.get(codigo_original, {})

        tipo = metadata.get("tipo") or _tipo_desde_archivo(case)
        incremento = metadata.get("incremento", "SIN-INCREMENTO")

        casos.append(
            {
                "codigo_original": codigo_original,
                "codigo_salida": metadata.get("codigo_salida", codigo_original),
                "tipo": tipo,
                "incremento": incremento,
                "nombre": metadata.get("nombre", _normalizar_nombre_test(nombre_original)),
                "objetivo": metadata.get("objetivo", "Sin descripción registrada."),
                "estado": _estado(case),
                "ms": _tiempo_ms(case),
                "nombre_original": nombre_original,
            }
        )

    return casos


def _filtrar(casos: list[dict], incremento: str, tipo: str) -> list[dict]:
    return [
        caso
        for caso in casos
        if caso["incremento"] == incremento and caso["tipo"] == tipo
    ]


def _contar(casos: list[dict], estado: str | None = None, tipo: str | None = None) -> int:
    filtrados = casos

    if estado is not None:
        filtrados = [caso for caso in filtrados if caso["estado"] == estado]

    if tipo is not None:
        filtrados = [caso for caso in filtrados if caso["tipo"] == tipo]

    return len(filtrados)


def _imprimir_grupo(titulo: str, casos: list[dict]) -> None:
    if not casos:
        return

    print(titulo)

    for caso in sorted(casos, key=_orden_salida):
        print(_linea_estado(caso["codigo_salida"], caso["nombre"], caso["estado"], caso["ms"]))
        #print(f"      Objetivo: {caso['objetivo']}")

    print()


def main() -> int:
    xml_path = Path(sys.argv[1] if len(sys.argv) > 1 else "reportes/pytest_resultados.xml")

    if not xml_path.exists():
        print(f"No se encontró el archivo XML: {xml_path}")
        return 1

    casos = _leer_casos(xml_path)

    aprobadas = _contar(casos, "APROBADA")
    no_aprobadas = sum(1 for caso in casos if caso["estado"] in {"NO APROBADA", "ERROR"})
    omitidas = _contar(casos, "OMITIDA")

    unitarias = _contar(casos, tipo="unitaria")
    integracion = _contar(casos, tipo="integracion")
    total = len(casos)

    total_ms = sum(caso["ms"] for caso in casos)

    print("======================================================================")
    print(" RESUMEN DE EJECUCIÓN DE PRUEBAS AUTOMATIZADAS - SEÑAPP")
    print("======================================================================\n")

    print("Entorno de ejecución:")
    print("  Herramienta: pytest")
    print(f"  Plataforma: {platform.system()}")
    print("  Base de datos: SQLite temporal")
    print("  Datos de prueba: fixtures basados en los datos iniciales de la base de datos")

    for incremento_id in ["I1", "I2", "I3", "I4"]:
        print("----------------------------------------------------------------------")
        print(INCREMENTOS[incremento_id])
        print("----------------------------------------------------------------------")

        _imprimir_grupo(
            "Pruebas unitarias",
            _filtrar(casos, incremento_id, "unitaria"),
        )
        _imprimir_grupo(
            "Pruebas de integración",
            _filtrar(casos, incremento_id, "integracion"),
        )

    otros = [caso for caso in casos if caso["incremento"] == "SIN-INCREMENTO"]
    if otros:
        print("----------------------------------------------------------------------")
        print("Pruebas no asociadas a incremento")
        print("----------------------------------------------------------------------")
        _imprimir_grupo("Casos no clasificados", otros)

    print("======================================================================")
    print("RESULTADO GENERAL")
    print("======================================================================")
    print(f"  Pruebas unitarias ejecutadas:       {unitarias:>2}")
    print(f"  Pruebas de integración ejecutadas:  {integracion:>2}")
    print(f"  Total de pruebas ejecutadas:        {total:>2}")
    print(f"  Aprobadas:                          {aprobadas:>2}")
    print(f"  No aprobadas:                       {no_aprobadas:>2}")

    if omitidas:
        print(f"  Omitidas:                           {omitidas:>2}")

    print(f"  Tiempo total informado:             {total_ms / 1000:.2f} s")

    estado_final = "EJECUCIÓN APROBADA" if no_aprobadas == 0 else "EJECUCIÓN CON OBSERVACIONES"
    print(f"\nEstado final: {estado_final}")
    print("======================================================================")

    return 0 if no_aprobadas == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
