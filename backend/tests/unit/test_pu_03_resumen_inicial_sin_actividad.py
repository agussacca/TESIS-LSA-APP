from app.services.estadisticas import calcular_resumen_estadisticas


def test_pu_03_resumen_inicial_sin_actividad():
    resumen = calcular_resumen_estadisticas(
        intentos=[],
        palabras_deletreadas=[],
        rondas=[],
    )

    assert resumen["senias_aprendidas_camara"] == 0
    assert resumen["palabras_deletreadas_exitosamente"] == 0
    assert resumen["rondas_por_categoria"] == {}
    assert resumen["progreso_por_letra"] == []
