from app.db import models
from tests.conftest import obtener_pk


def test_pi_07_registro_ronda_actualizacion_estadisticas(app_cliente, db_session, usuario_registrado, categorias_base):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")
    deportes_id = obtener_pk(categorias_base["deportes"], "id_categoria_aprendizaje", "id")

    response = app_cliente.post(
        "/api/rondas-minijuego",
        json={
            "usuario_id": usuario_id,
            "categoria_id": deportes_id,
            "cantidad_minijuegos": 5,
            "correctas": 5,
        },
    )

    assert response.status_code == 200

    ronda = db_session.query(models.RondaMinijuegoUsuario).first()
    assert ronda is not None
    assert ronda.cantidad_minijuegos == 5
    assert ronda.correctas == 5
    assert ronda.correctas == ronda.cantidad_minijuegos
