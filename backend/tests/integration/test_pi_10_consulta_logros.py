from app.db import models
from tests.conftest import obtener_pk


def test_pi_10_consulta_logros(app_cliente, db_session, usuario_registrado):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")

    logro = models.Logro(
        nombre="Primer paso",
        descripcion="Realizar la primera práctica aceptada.",
        imagen_url="/static/logros/primer_paso.png",
    )
    db_session.add(logro)
    db_session.flush()

    logro_id = obtener_pk(logro, "id_logro", "id")
    db_session.add(models.LogroDesbloqueadoUsuario(usuario_id=usuario_id, logro_id=logro_id))
    db_session.commit()

    response = app_cliente.get(f"/api/logros-usuario/{usuario_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["desbloqueados"] == 1
    assert data["logros"][0]["nombre"] == "Primer paso"
    assert data["logros"][0]["fecha_desbloqueo"] is not None
