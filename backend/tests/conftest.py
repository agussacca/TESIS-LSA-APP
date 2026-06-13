from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT

# Permite ubicar el paquete app si las pruebas se copian dentro de backend/tests
# o si se copian en una carpeta tests ubicada en la raíz del proyecto.
if not (BACKEND_DIR / "app").exists() and (ROOT / "backend" / "app").exists():
    BACKEND_DIR = ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture()
def engine_temporal():
    from app.db.database import Base

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def db_session(engine_temporal):
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine_temporal,
    )
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def app_cliente(db_session):
    from app.main import app
    from app.db.database import get_db

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def obtener_pk(obj: Any, *candidatos: str) -> int:
    for nombre in candidatos:
        if hasattr(obj, nombre):
            valor = getattr(obj, nombre)
            if valor is not None:
                return valor
    raise AssertionError(f"No se encontró clave primaria en {obj!r}. Candidatos: {candidatos}")


@pytest.fixture()
def usuario_registrado(db_session):
    from app.db import models

    usuario = models.Usuario(
        email="usuario.prueba@senapp.test",
        password_hash="hash_de_prueba",
        nombre_visible="Usuario de prueba",
        foto_perfil_url=None,
    )
    db_session.add(usuario)
    db_session.flush()

    usuario_id = obtener_pk(usuario, "id_usuario", "id")

    if hasattr(models, "ProgresoUsuario"):
        progreso = models.ProgresoUsuario(
            usuario_id=usuario_id,
            xp_total=0,
            nivel=1,
            racha_actual=0,
            racha_maxima=0,
        )
        db_session.add(progreso)

    db_session.commit()
    db_session.refresh(usuario)
    return usuario


@pytest.fixture()
def categorias_base(db_session):
    from app.db import models

    abecedario = models.CategoriaAprendizaje(
        nombre="Abecedario",
        descripcion="Señas del abecedario",
        imagen_portada_url="/static/categorias/abecedario.png",
        orden=1,
    )
    deportes = models.CategoriaAprendizaje(
        nombre="Deportes",
        descripcion="Señas asociadas a deportes",
        imagen_portada_url="/static/categorias/deportes.png",
        orden=2,
    )
    db_session.add_all([abecedario, deportes])
    db_session.flush()

    abecedario_id = obtener_pk(abecedario, "id_categoria_aprendizaje", "id")
    deportes_id = obtener_pk(deportes, "id_categoria_aprendizaje", "id")

    senias = [
        models.Senia(
            categoria_id=abecedario_id,
            nombre="A",
            descripcion="Letra A en LSA",
            imagen_url="/static/senias/a.png",
            video_url="/static/senias/a.mp4",
            orden=1,
        ),
        models.Senia(
            categoria_id=abecedario_id,
            nombre="B",
            descripcion="Letra B en LSA",
            imagen_url="/static/senias/b.png",
            video_url="/static/senias/b.mp4",
            orden=2,
        ),
        models.Senia(
            categoria_id=deportes_id,
            nombre="Fútbol",
            descripcion="Seña de fútbol",
            imagen_url="/static/senias/futbol.png",
            video_url="/static/senias/futbol.mp4",
            orden=1,
        ),
    ]
    db_session.add_all(senias)
    db_session.commit()

    return {"abecedario": abecedario, "deportes": deportes}


@pytest.fixture()
def personalizacion_base(db_session):
    from app.db import models

    marcos = [
        models.Marco(nombre="Fuego", imagen_url="/static/marcos/fuego.png", nivel_requerido=1, orden=1),
        models.Marco(nombre="Madera", imagen_url="/static/marcos/madera.png", nivel_requerido=2, orden=2),
        models.Marco(nombre="Fútbol", imagen_url="/static/marcos/futbol.png", nivel_requerido=None, orden=20),
    ]
    titulos = [
        models.Titulo(nombre="Aprendiz constante", nivel_requerido=1, orden=1),
        models.Titulo(nombre="Maestro de señas", nivel_requerido=10, orden=10),
    ]
    db_session.add_all(marcos + titulos)
    db_session.commit()
    return {"marcos": marcos, "titulos": titulos}
