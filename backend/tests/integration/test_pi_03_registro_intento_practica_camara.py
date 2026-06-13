from app.db import models
from tests.conftest import obtener_pk


def test_pi_03_registro_intento_practica_camara(app_cliente, db_session, usuario_registrado):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")

    response = app_cliente.post(
        "/api/intentos-practica",
        json={
            "usuario_id": usuario_id,
            "letra_esperada": "A",
            "letra_predicha": "A",
            "validado": True,
        },
    )

    assert response.status_code == 200

    intento = db_session.query(models.IntentoPractica).first()
    assert intento is not None
    assert intento.letra_esperada == "A"
    assert intento.letra_predicha == "A"
    assert intento.validado is True
