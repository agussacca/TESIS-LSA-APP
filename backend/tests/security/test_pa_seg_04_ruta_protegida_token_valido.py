def test_pa_seg_04_ruta_protegida_token_valido(app_cliente, usuario_registrado):
    login = app_cliente.post(
        "/api/auth/login",
        json={
            "email": usuario_registrado.email,
            "password": "hash_de_prueba",
        },
    )
    assert login.status_code == 200

    token = login.json()["access_token"]

    response = app_cliente.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["usuario"]["id_usuario"] == usuario_registrado.id_usuario
    assert data["usuario"]["email"] == usuario_registrado.email
