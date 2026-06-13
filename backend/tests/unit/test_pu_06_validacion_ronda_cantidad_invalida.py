import pytest

from app.services.estadisticas import validar_ronda_minijuego


def test_pu_06_validacion_ronda_cantidad_invalida():
    with pytest.raises(ValueError):
        validar_ronda_minijuego(cantidad_minijuegos=-1, correctas=0)

    with pytest.raises(ValueError):
        validar_ronda_minijuego(cantidad_minijuegos=3, correctas=4)

    validar_ronda_minijuego(cantidad_minijuegos=5, correctas=3)
