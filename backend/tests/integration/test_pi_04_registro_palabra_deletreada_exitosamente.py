from app.db import models
from tests.conftest import obtener_pk


def test_pi_04_registro_palabra_deletreada_exitosamente(app_cliente, db_session, usuario_registrado):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")

    response = app_cliente.post(
        "/api/palabras-deletreadas",
        json={
            "usuario_id": usuario_id,
            "palabra": "CASA",
        },
    )

    assert response.status_code == 200

    palabra = db_session.query(models.PalabraDeletreadaUsuario).first()
    assert palabra is not None
    assert palabra.palabra == "CASA"
