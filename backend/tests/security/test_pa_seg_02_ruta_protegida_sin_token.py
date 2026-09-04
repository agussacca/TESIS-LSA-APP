def test_pa_seg_02_ruta_protegida_sin_token(app_cliente):
    response = app_cliente.get("/api/auth/me")

    assert response.status_code == 401

    data = response.json()
    assert "detail" in data
    assert "usuario" not in data
