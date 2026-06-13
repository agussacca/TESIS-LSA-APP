from app.db import models
from tests.conftest import obtener_pk


def test_pi_09_objetivo_mediante_ronda_minijuego(app_cliente, db_session, usuario_registrado, categorias_base):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")
    deportes_id = obtener_pk(categorias_base["deportes"], "id_categoria_aprendizaje", "id")

    objetivo = models.Objetivo(
        nombre="Completar una ronda",
        descripcion="Completar una ronda de minijuegos en el día.",
        periodicidad="DIARIA",
        xp_recompensa=20,
    )
    db_session.add(objetivo)
    db_session.commit()

    app_cliente.post("/api/rondas-minijuego", json={
        "usuario_id": usuario_id,
        "categoria_id": deportes_id,
        "cantidad_minijuegos": 5,
        "correctas": 3,
    })

    response = app_cliente.post(f"/api/gamificacion/sincronizar/{usuario_id}")

    assert response.status_code == 200
    completados = db_session.query(models.ObjetivoCompletadoUsuario).all()
    assert len(completados) == 1
