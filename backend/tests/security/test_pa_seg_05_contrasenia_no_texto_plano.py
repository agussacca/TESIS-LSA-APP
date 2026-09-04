from app.core.security import verify_password
from app.db import models


def test_pa_seg_05_contrasenia_no_texto_plano(app_cliente, db_session):
    email = "seguridad.password@senapp.test"
    password_original = "ClaveSegura123"

    response = app_cliente.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": password_original,
            "nombre_visible": "Usuario seguridad",
            "foto_perfil_url": None,
        },
    )

    assert response.status_code == 200

    usuario = (
        db_session.query(models.Usuario)
        .filter(models.Usuario.email == email)
        .one()
    )

    assert usuario.password_hash != password_original
    assert usuario.password_hash.startswith("pbkdf2_sha256$")
    assert verify_password(password_original, usuario.password_hash) is True
