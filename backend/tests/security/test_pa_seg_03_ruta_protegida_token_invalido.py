def test_pa_seg_03_ruta_protegida_token_invalido(app_cliente, usuario_registrado):
    login = app_cliente.post(
        "/api/auth/login",
        json={
            "email": usuario_registrado.email,
            "password": "hash_de_prueba",
        },
    )
    assert login.status_code == 200

    token_valido = login.json()["access_token"]
    header, payload, signature = token_valido.split(".")

    primer_caracter_alterado = "A" if signature[0] != "A" else "B"
    signature_alterada = primer_caracter_alterado + signature[1:]
    token_invalido = f"{header}.{payload}.{signature_alterada}"

    response = app_cliente.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token_invalido}"},
    )

    assert response.status_code == 401

    data = response.json()
    assert "detail" in data
    assert "usuario" not in data
