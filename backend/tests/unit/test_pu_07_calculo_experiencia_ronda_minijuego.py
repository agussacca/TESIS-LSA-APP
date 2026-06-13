from app.services.gamificacion import calcular_xp_ronda_minijuego


def test_pu_07_calculo_experiencia_ronda_minijuego():
    xp_parcial = calcular_xp_ronda_minijuego(cantidad_minijuegos=5, correctas=3)
    xp_perfecta = calcular_xp_ronda_minijuego(cantidad_minijuegos=5, correctas=5)

    assert xp_parcial == 15
    assert xp_perfecta == 35
    assert xp_perfecta > xp_parcial
