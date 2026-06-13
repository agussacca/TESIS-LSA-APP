
def test_pi_01_consulta_contenido_educativo_disponible(app_cliente, categorias_base):
    response = app_cliente.get("/api/contenido-aprendizaje")

    assert response.status_code == 200
    data = response.json()

    assert len(data) >= 2
    assert [categoria["nombre"] for categoria in data[:2]] == ["Abecedario", "Deportes"]
    assert data[0]["senias"][0]["nombre"] == "A"
    assert "imagen_url" in data[0]["senias"][0]
    assert "video_url" in data[0]["senias"][0]
