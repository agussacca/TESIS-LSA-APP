from datetime import date

from app.services.gamificacion import actualizar_racha_por_objetivo_diario


def test_pu_09_actualizacion_racha_diaria():
    inicial = actualizar_racha_por_objetivo_diario(
        fecha_ultima_racha=None,
        racha_actual=0,
        racha_maxima=0,
        fecha_actual=date(2026, 6, 10),
    )
    assert inicial["racha_actual"] == 1
    assert inicial["racha_maxima"] == 1

    consecutiva = actualizar_racha_por_objetivo_diario(
        fecha_ultima_racha=date(2026, 6, 10),
        racha_actual=1,
        racha_maxima=1,
        fecha_actual=date(2026, 6, 11),
    )
    assert consecutiva["racha_actual"] == 2
    assert consecutiva["racha_maxima"] == 2

    interrumpida = actualizar_racha_por_objetivo_diario(
        fecha_ultima_racha=date(2026, 6, 10),
        racha_actual=2,
        racha_maxima=2,
        fecha_actual=date(2026, 6, 12),
    )
    assert interrumpida["racha_actual"] == 1
    assert interrumpida["racha_maxima"] == 2
