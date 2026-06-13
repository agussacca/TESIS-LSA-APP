from app.services.estadisticas import calcular_estadisticas_practica_por_letra


def test_pu_04_calculo_estadisticas_practica_por_letra():
    intentos = [
        {"letra_esperada": "A", "letra_predicha": "A", "validado": True},
        {"letra_esperada": "A", "letra_predicha": "A", "validado": False},
        {"letra_esperada": "A", "letra_predicha": "B", "validado": True},
        {"letra_esperada": "B", "letra_predicha": "B", "validado": True},
    ]

    estadisticas = calcular_estadisticas_practica_por_letra(intentos)

    assert estadisticas["A"]["total_intentos"] == 3
    assert estadisticas["A"]["intentos_aceptados"] == 1
    assert estadisticas["A"]["precision"] == 1 / 3

    assert estadisticas["B"]["total_intentos"] == 1
    assert estadisticas["B"]["intentos_aceptados"] == 1
    assert estadisticas["B"]["precision"] == 1.0
