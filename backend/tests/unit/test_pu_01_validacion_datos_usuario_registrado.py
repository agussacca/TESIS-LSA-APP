import pytest
from pydantic import ValidationError

from app.db import schemas


def test_pu_01_validacion_datos_usuario_registrado():
    usuario = schemas.UsuarioCrear(
        email="persona@senapp.test",
        password="ClaveSegura123",
        nombre_visible="Persona de prueba",
        foto_perfil_url=None,
    )

    assert usuario.email == "persona@senapp.test"
    assert usuario.nombre_visible == "Persona de prueba"

    with pytest.raises(ValidationError):
        schemas.UsuarioCrear(
            email="correo_invalido",
            password="123",
            nombre_visible="",
        )
