from tests.conftest import obtener_pk


def test_pi_05_registro_intentos_consulta_progreso_letra(app_cliente, usuario_registrado):
    usuario_id = obtener_pk(usuario_registrado, "id_usuario", "id")

    payloads = [
        {"usuario_id": usuario_id, "letra_esperada": "A", "letra_predicha": "A", "validado": True},
        {"usuario_id": usuario_id, "letra_esperada": "A", "letra_predicha": "B", "validado": True},
        {"usuario_id": usuario_id, "letra_esperada": "B", "letra_predicha": "B", "validado": False},
    ]

    for payload in payloads:
        assert app_cliente.post("/api/intentos-practica", json=payload).status_code == 200

    response = app_cliente.get(f"/api/progreso-letras/{usuario_id}")

    assert response.status_code == 200
    data = response.json()
    por_letra = {fila["letra"]: fila for fila in data}

    assert por_letra["A"]["total_intentos"] == 2
    assert por_letra["A"]["intentos_aceptados"] == 1
    assert por_letra["B"]["total_intentos"] == 1
    assert por_letra["B"]["intentos_aceptados"] == 0
