from app.services.estadisticas import agrupar_progreso_por_letra


def test_pu_05_agrupacion_progreso_por_letra():
    intentos = [
        {"letra_esperada": "C", "letra_predicha": "C", "validado": True},
        {"letra_esperada": "A", "letra_predicha": "A", "validado": True},
        {"letra_esperada": "C", "letra_predicha": "D", "validado": True},
    ]

    progreso = agrupar_progreso_por_letra(intentos)

    assert [fila["letra"] for fila in progreso] == ["A", "C"]
    assert progreso[0]["total_intentos"] == 1
    assert progreso[0]["intentos_aceptados"] == 1
    assert progreso[1]["total_intentos"] == 2
    assert progreso[1]["intentos_aceptados"] == 1
