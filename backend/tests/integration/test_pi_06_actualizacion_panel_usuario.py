from tests.conftest import obtener_pk


def test_pi_06_actualizacion_panel_usuario(app_cliente, usuario_registrado, categorias_base):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")
    deportes_id = obtener_pk(categorias_base["deportes"], "id_categoria_aprendizaje", "id")

    app_cliente.post("/api/intentos-practica", json={
        "usuario_id": usuario_id,
        "letra_esperada": "A",
        "letra_predicha": "A",
        "validado": True,
    })
    app_cliente.post("/api/palabras-deletreadas", json={
        "usuario_id": usuario_id,
        "palabra": "CASA",
    })
    app_cliente.post("/api/rondas-minijuego", json={
        "usuario_id": usuario_id,
        "categoria_id": deportes_id,
        "cantidad_minijuegos": 5,
        "correctas": 5,
    })

    response = app_cliente.get(f"/api/panel-usuario/{usuario_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["senias_aprendidas_camara"] == 1
    assert data["palabras_deletreadas_exitosamente"] == 1
    assert data["rondas_por_categoria"]["Deportes"] >= 1
