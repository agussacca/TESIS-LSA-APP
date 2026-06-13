from app.db import models


def test_pi_02_no_persistencia_practica_invitado(app_cliente, db_session):
    total_antes = db_session.query(models.IntentoPractica).count()

    response = app_cliente.post(
        "/api/intentos-practica",
        json={
            "letra_esperada": "A",
            "letra_predicha": "A",
            "validado": True,
        },
    )

    assert response.status_code in {400, 401, 403, 422}
    assert db_session.query(models.IntentoPractica).count() == total_antes
