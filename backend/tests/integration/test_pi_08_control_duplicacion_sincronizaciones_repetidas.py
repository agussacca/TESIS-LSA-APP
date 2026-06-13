from tests.conftest import obtener_pk


def test_pi_08_control_duplicacion_sincronizaciones_repetidas(app_cliente, usuario_registrado):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")

    primera = app_cliente.post(f"/api/gamificacion/sincronizar/{usuario_id}")
    segunda = app_cliente.post(f"/api/gamificacion/sincronizar/{usuario_id}")

    assert primera.status_code == 200
    assert segunda.status_code == 200

    data_1 = primera.json()
    data_2 = segunda.json()

    assert data_2["progreso"]["xp_total"] == data_1["progreso"]["xp_total"]
    assert len(data_2.get("objetivos_completados", [])) == 0
    assert len(data_2.get("logros_desbloqueados", [])) == 0
