from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UsuarioCrear(BaseModel):
    email: str
    password: str
    nombre_visible: str
    foto_perfil_url: str | None = None

    @field_validator("email")
    @classmethod
    def validar_email(cls, value: str) -> str:
        value = (value or "").strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
            raise ValueError("El correo electrónico no tiene un formato válido.")
        return value

    @field_validator("password")
    @classmethod
    def validar_password(cls, value: str) -> str:
        if value is None or len(value) < 8:
            raise ValueError("La contraseña debe tener al menos 8 caracteres.")
        return value

    @field_validator("nombre_visible")
    @classmethod
    def validar_nombre_visible(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("El nombre visible es obligatorio.")
        return value


class UsuarioLogin(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validar_email(cls, value: str) -> str:
        value = (value or "").strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
            raise ValueError("El correo electrónico no tiene un formato válido.")
        return value

    @field_validator("password")
    @classmethod
    def validar_password(cls, value: str) -> str:
        if not value:
            raise ValueError("La contraseña es obligatoria.")
        return value


class AuthTokenRespuesta(BaseModel):
    access_token: str
    token_type: str = "bearer"
    usuario: "UsuarioRespuesta"


class UsuarioActualizar(BaseModel):
    usuario_id: int
    email: str | None = None
    nombre_visible: str | None = None
    foto_perfil_url: str | None = None


class UsuarioRespuesta(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id_usuario: int
    email: str
    nombre_visible: str
    foto_perfil_url: str | None = None
    marco_equipado_id: int | None = None
    titulo_equipado_id: int | None = None


class SeniaRespuesta(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id_senia: int
    nombre: str
    descripcion: str | None = None
    imagen_url: str | None = None
    video_url: str | None = None
    orden: int


class CategoriaConSeniasRespuesta(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id_categoria_aprendizaje: int
    nombre: str
    descripcion: str | None = None
    imagen_portada_url: str | None = None
    orden: int
    senias: list[SeniaRespuesta] = Field(default_factory=list)


class IntentoPracticaCrear(BaseModel):
    usuario_id: int
    letra_esperada: str
    letra_predicha: str
    validado: bool

    @field_validator("letra_esperada", "letra_predicha")
    @classmethod
    def normalizar_letra(cls, value: str) -> str:
        value = (value or "").strip().upper()
        if not value:
            raise ValueError("La letra es obligatoria.")
        return value


class PalabraDeletreadaCrear(BaseModel):
    usuario_id: int
    palabra: str

    @field_validator("palabra")
    @classmethod
    def normalizar_palabra(cls, value: str) -> str:
        value = (value or "").strip().upper()
        if not value:
            raise ValueError("La palabra es obligatoria.")
        return value


class RondaMinijuegoCrear(BaseModel):
    usuario_id: int
    categoria_id: int
    cantidad_minijuegos: int = Field(..., ge=1)
    correctas: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validar_consistencia(self) -> "RondaMinijuegoCrear":
        if self.correctas > self.cantidad_minijuegos:
            raise ValueError("Las respuestas correctas no pueden superar la cantidad total de minijuegos.")
        return self


class EquipamientoPerfilCrear(BaseModel):
    usuario_id: int
    marco_id: int
    titulo_id: int


class ProgresoRespuesta(BaseModel):
    xp_total: int
    nivel: int
    xp_nivel_actual: int = 0
    xp_siguiente_nivel: int = 120
    racha_actual: int
    racha_maxima: int
    fecha_ultima_racha: Any | None = None


class ObjetivoUsuarioRespuesta(BaseModel):
    id: int
    codigo: str
    titulo: str
    descripcion: str
    actual: int
    objetivo: int
    xp: int
    completado: bool
    xp_otorgado: bool = False


class ObjetivosUsuarioRespuesta(BaseModel):
    diarios: list[ObjetivoUsuarioRespuesta]
    semanales: list[ObjetivoUsuarioRespuesta]
    progreso: ProgresoRespuesta
    eventos: list[dict[str, Any]] = Field(default_factory=list)


class LogroUsuarioRespuesta(BaseModel):
    id_logro: int
    id: int
    codigo: str
    familia: str = "General"
    nombre: str
    descripcion: str
    imagen_url: str | None = None
    orden: int = 0
    desbloqueado: bool
    fecha_desbloqueo: datetime | None = None


class LogrosUsuarioRespuesta(BaseModel):
    total: int
    desbloqueados: int
    pendientes: int
    logros: list[LogroUsuarioRespuesta]


class MarcoRespuesta(BaseModel):
    id_marco: int
    id: int
    nombre: str
    imagen_url: str
    nivel_requerido: int | None = None
    orden: int
    disponible: bool | None = None


class TituloRespuesta(BaseModel):
    id_titulo: int
    id: int
    nombre: str
    nivel_requerido: int
    orden: int
    disponible: bool | None = None


try:
    AuthTokenRespuesta.model_rebuild()
except Exception:
    pass
