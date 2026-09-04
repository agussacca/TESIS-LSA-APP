def test_pa_seg_01_login_valido_genera_token(app_cliente, usuario_registrado):
    response = app_cliente.post(
        "/api/auth/login",
        json={
            "email": usuario_registrado.email,
            "password": "hash_de_prueba",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data.get("access_token"), str)
    assert data["access_token"].count(".") == 2
    assert data["token_type"] == "bearer"
    assert data["usuario"]["email"] == usuario_registrado.email
