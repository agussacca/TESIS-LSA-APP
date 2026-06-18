from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import create_access_token, get_current_user
from app.db import crud, models, schemas
from app.db.database import get_db

router = APIRouter(prefix="/api/auth", tags=["Autenticación"])


def _auth_response(db: Session, usuario: models.Usuario) -> dict:
    token = create_access_token(
        subject=usuario.id_usuario,
        extra_claims={"email": usuario.email},
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "usuario": crud.consultar_usuario_registrado(db, usuario.id_usuario),
    }


@router.post("/register")
def registrar_usuario(data: schemas.UsuarioCrear, db: Session = Depends(get_db)):
    usuario = crud.crear_usuario_registrado(db, data)
    return _auth_response(db, usuario)


@router.post("/login")
def iniciar_sesion(data: schemas.UsuarioLogin, db: Session = Depends(get_db)):
    usuario = crud.autenticar_usuario(db, data.email, data.password)
    if usuario is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo electrónico o contraseña incorrectos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _auth_response(db, usuario)


@router.get("/me")
def obtener_sesion_actual(usuario: models.Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    return {
        "usuario": crud.consultar_usuario_registrado(db, usuario.id_usuario),
    }


@router.post("/logout")
def cerrar_sesion():
    # Con JWT sin tabla de sesiones, cerrar sesión consiste en eliminar el token del cliente.
    return {"ok": True}
