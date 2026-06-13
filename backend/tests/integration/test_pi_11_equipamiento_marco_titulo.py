from tests.conftest import obtener_pk


def test_pi_11_equipamiento_marco_titulo(app_cliente, usuario_registrado, personalizacion_base):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")
    marco_fuego_id = obtener_pk(personalizacion_base["marcos"][0], "id_marco", "id")
    titulo_base_id = obtener_pk(personalizacion_base["titulos"][0], "id_titulo", "id")

    response = app_cliente.post(
        "/api/perfil/equipamiento",
        json={
            "usuario_id": usuario_id,
            "marco_id": marco_fuego_id,
            "titulo_id": titulo_base_id,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["marco_equipado"]["nombre"] == "Fuego"
    assert data["titulo_equipado"]["nombre"] == "Aprendiz constante"
