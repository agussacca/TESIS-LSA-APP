from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Usuario(Base):
    __tablename__ = "usuarios"

    id_usuario: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    email: Mapped[str] = mapped_column(String, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    nombre_visible: Mapped[str] = mapped_column(String, nullable=False)
    foto_perfil_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    marco_equipado_id: Mapped[Optional[int]] = mapped_column(ForeignKey("marcos.id_marco"), nullable=True)
    titulo_equipado_id: Mapped[Optional[int]] = mapped_column(ForeignKey("titulos.id_titulo"), nullable=True)

    marco_equipado: Mapped[Optional["Marco"]] = relationship("Marco", foreign_keys=[marco_equipado_id])
    titulo_equipado: Mapped[Optional["Titulo"]] = relationship("Titulo", foreign_keys=[titulo_equipado_id])

    intentos_practica: Mapped[list["IntentoPractica"]] = relationship(
        back_populates="usuario",
        cascade="all, delete-orphan",
    )
    palabras_deletreadas: Mapped[list["PalabraDeletreadaUsuario"]] = relationship(
        back_populates="usuario",
        cascade="all, delete-orphan",
    )
    rondas_minijuego: Mapped[list["RondaMinijuegoUsuario"]] = relationship(
        back_populates="usuario",
        cascade="all, delete-orphan",
    )
    progreso: Mapped[Optional["ProgresoUsuario"]] = relationship(
        back_populates="usuario",
        uselist=False,
        cascade="all, delete-orphan",
    )
    objetivos_completados: Mapped[list["ObjetivoCompletadoUsuario"]] = relationship(
        back_populates="usuario",
        cascade="all, delete-orphan",
    )
    logros_desbloqueados: Mapped[list["LogroDesbloqueadoUsuario"]] = relationship(
        back_populates="usuario",
        cascade="all, delete-orphan",
    )


class CategoriaAprendizaje(Base):
    __tablename__ = "categorias_aprendizaje"

    id_categoria_aprendizaje: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    nombre: Mapped[str] = mapped_column(String, nullable=False, index=True)
    descripcion: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    imagen_portada_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    orden: Mapped[int] = mapped_column(Integer, nullable=False)

    senias: Mapped[list["Senia"]] = relationship(
        back_populates="categoria",
        cascade="all, delete-orphan",
    )
    rondas: Mapped[list["RondaMinijuegoUsuario"]] = relationship(back_populates="categoria")


class Senia(Base):
    __tablename__ = "senias"

    id_senia: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    categoria_id: Mapped[int] = mapped_column(
        ForeignKey("categorias_aprendizaje.id_categoria_aprendizaje"),
        nullable=False,
        index=True,
    )
    nombre: Mapped[str] = mapped_column(String, nullable=False)
    descripcion: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    imagen_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    video_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    orden: Mapped[int] = mapped_column(Integer, nullable=False)

    categoria: Mapped["CategoriaAprendizaje"] = relationship(back_populates="senias")


class IntentoPractica(Base):
    __tablename__ = "intentos_practica"

    id_intento_practica: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id_usuario"), nullable=False, index=True)
    letra_esperada: Mapped[str] = mapped_column(String, nullable=False)
    letra_predicha: Mapped[str] = mapped_column(String, nullable=False)
    validado: Mapped[bool] = mapped_column(nullable=False, default=False)

    usuario: Mapped["Usuario"] = relationship(back_populates="intentos_practica")


class PalabraDeletreadaUsuario(Base):
    __tablename__ = "palabras_deletreadas_usuario"

    id_palabra_deletreada_usuario: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id_usuario"), nullable=False, index=True)
    palabra: Mapped[str] = mapped_column(String, nullable=False)

    usuario: Mapped["Usuario"] = relationship(back_populates="palabras_deletreadas")


class RondaMinijuegoUsuario(Base):
    __tablename__ = "rondas_minijuego_usuario"

    id_ronda_minijuego_usuario: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id_usuario"), nullable=False, index=True)
    categoria_id: Mapped[int] = mapped_column(
        ForeignKey("categorias_aprendizaje.id_categoria_aprendizaje"),
        nullable=False,
        index=True,
    )
    cantidad_minijuegos: Mapped[int] = mapped_column(Integer, nullable=False)
    correctas: Mapped[int] = mapped_column(Integer, nullable=False)

    usuario: Mapped["Usuario"] = relationship(back_populates="rondas_minijuego")
    categoria: Mapped["CategoriaAprendizaje"] = relationship(back_populates="rondas")


class ProgresoUsuario(Base):
    __tablename__ = "progreso_usuario"

    id_progreso_usuario: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id_usuario"), nullable=False, unique=True, index=True)
    xp_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    nivel: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    racha_actual: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    racha_maxima: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fecha_ultima_racha: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    usuario: Mapped["Usuario"] = relationship(back_populates="progreso")


class Objetivo(Base):
    __tablename__ = "objetivos"

    id_objetivo: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    nombre: Mapped[str] = mapped_column(String, nullable=False)
    descripcion: Mapped[str] = mapped_column(Text, nullable=False)
    periodicidad: Mapped[str] = mapped_column(String, nullable=False)
    xp_recompensa: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    completados: Mapped[list["ObjetivoCompletadoUsuario"]] = relationship(
        back_populates="objetivo",
        cascade="all, delete-orphan",
    )


class ObjetivoCompletadoUsuario(Base):
    __tablename__ = "objetivos_completados_usuario"

    id_objetivo_completado_usuario: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id_usuario"), nullable=False, index=True)
    objetivo_id: Mapped[int] = mapped_column(ForeignKey("objetivos.id_objetivo"), nullable=False, index=True)
    clave_periodo: Mapped[str] = mapped_column(String, nullable=False, index=True)
    fecha_completado: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    usuario: Mapped["Usuario"] = relationship(back_populates="objetivos_completados")
    objetivo: Mapped["Objetivo"] = relationship(back_populates="completados")


class Logro(Base):
    __tablename__ = "logros"

    id_logro: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    nombre: Mapped[str] = mapped_column(String, nullable=False)
    descripcion: Mapped[str] = mapped_column(Text, nullable=False)
    imagen_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    desbloqueos: Mapped[list["LogroDesbloqueadoUsuario"]] = relationship(
        back_populates="logro",
        cascade="all, delete-orphan",
    )


class LogroDesbloqueadoUsuario(Base):
    __tablename__ = "logros_desbloqueados_usuario"

    id_logro_desbloqueado_usuario: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id_usuario"), nullable=False, index=True)
    logro_id: Mapped[int] = mapped_column(ForeignKey("logros.id_logro"), nullable=False, index=True)
    fecha_desbloqueo: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    usuario: Mapped["Usuario"] = relationship(back_populates="logros_desbloqueados")
    logro: Mapped["Logro"] = relationship(back_populates="desbloqueos")


class Marco(Base):
    __tablename__ = "marcos"

    id_marco: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    nombre: Mapped[str] = mapped_column(String, nullable=False, index=True)
    imagen_url: Mapped[str] = mapped_column(String, nullable=False)
    nivel_requerido: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    orden: Mapped[int] = mapped_column(Integer, nullable=False)


class Titulo(Base):
    __tablename__ = "titulos"

    id_titulo: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    nombre: Mapped[str] = mapped_column(String, nullable=False, index=True)
    nivel_requerido: Mapped[int] = mapped_column(Integer, nullable=False)
    orden: Mapped[int] = mapped_column(Integer, nullable=False)
