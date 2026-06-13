from app.services.gamificacion import objetivo_puede_otorgar_xp


def test_pu_10_prevencion_recompensa_duplicada_objetivo():
    completados = [
        {"objetivo_id": 1, "clave_periodo": "2026-06-10"},
    ]

    assert objetivo_puede_otorgar_xp(
        completados=completados,
        objetivo_id=2,
        clave_periodo="2026-06-10",
    ) is True

    assert objetivo_puede_otorgar_xp(
        completados=completados,
        objetivo_id=1,
        clave_periodo="2026-06-10",
    ) is False
