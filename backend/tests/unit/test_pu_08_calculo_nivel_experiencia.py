from app.services.gamificacion import calcular_nivel


def test_pu_08_calculo_nivel_experiencia():
    assert calcular_nivel(0) == 1
    assert calcular_nivel(1) == 1
    assert calcular_nivel(999999) == 10
