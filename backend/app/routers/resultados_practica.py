from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import crud, models, schemas
from app.db.database import get_db

router = APIRouter(prefix="/api", tags=["SeñApp"])


@router.post("/usuarios")
def crear_usuario(data: schemas.UsuarioCrear, db: Session = Depends(get_db)):
    usuario = crud.crear_usuario(db, data)
    return {
        "id_usuario": usuario.id_usuario,
        "email": usuario.email,
        "nombre_visible": usuario.nombre_visible,
        "foto_perfil_url": usuario.foto_perfil_url,
        "marco_equipado_id": usuario.marco_equipado_id,
        "titulo_equipado_id": usuario.titulo_equipado_id,
    }


@router.get("/usuarios/{usuario_id}")
def obtener_usuario_registrado(usuario_id: int, db: Session = Depends(get_db)):
    return crud.consultar_usuario_registrado(db, usuario_id)


@router.get("/contenido-aprendizaje")
def obtener_contenido_aprendizaje(db: Session = Depends(get_db)):
    categorias = crud.listar_contenido_aprendizaje(db)
    salida = []
    for categoria in categorias:
        senias = sorted(categoria.senias, key=lambda item: item.orden)
        salida.append(
            {
                "id_categoria_aprendizaje": categoria.id_categoria_aprendizaje,
                "id": categoria.id_categoria_aprendizaje,
                "nombre": categoria.nombre,
                "descripcion": categoria.descripcion,
                "imagen_portada_url": categoria.imagen_portada_url,
                "orden": categoria.orden,
                "senias": [
                    {
                        "id_senia": senia.id_senia,
                        "id": senia.id_senia,
                        "categoria_id": senia.categoria_id,
                        "nombre": senia.nombre,
                        "descripcion": senia.descripcion,
                        "imagen_url": senia.imagen_url,
                        "video_url": senia.video_url,
                        "orden": senia.orden,
                    }
                    for senia in senias
                ],
            }
        )
    return salida


@router.post("/intentos-practica")
def registrar_intento_practica(data: schemas.IntentoPracticaCrear, db: Session = Depends(get_db)):
    intento = crud.crear_intento_practica(db, data)
    return {
        "id_intento_practica": intento.id_intento_practica,
        "usuario_id": intento.usuario_id,
        "letra_esperada": intento.letra_esperada,
        "letra_predicha": intento.letra_predicha,
        "validado": intento.validado,
    }


@router.post("/palabras-deletreadas")
def registrar_palabra_deletreada(data: schemas.PalabraDeletreadaCrear, db: Session = Depends(get_db)):
    palabra = crud.crear_palabra_deletreada(db, data)
    return {
        "id_palabra_deletreada_usuario": palabra.id_palabra_deletreada_usuario,
        "usuario_id": palabra.usuario_id,
        "palabra": palabra.palabra,
    }


@router.get("/progreso-letras/{usuario_id}")
def obtener_progreso_letras(usuario_id: int, db: Session = Depends(get_db)):
    return crud.obtener_progreso_por_letra(db, usuario_id)


@router.get("/panel-usuario/{usuario_id}")
def obtener_panel_usuario(usuario_id: int, db: Session = Depends(get_db)):
    return crud.obtener_panel_usuario(db, usuario_id)


@router.get("/objetivos-usuario/{usuario_id}")
def obtener_objetivos_usuario(usuario_id: int, db: Session = Depends(get_db)):
    return crud.consultar_objetivos_usuario(db, usuario_id)


@router.post("/rondas-minijuego")
def registrar_ronda_minijuego(data: schemas.RondaMinijuegoCrear, db: Session = Depends(get_db)):
    ronda = crud.crear_ronda_minijuego(db, data)
    return {
        "id_ronda_minijuego_usuario": ronda.id_ronda_minijuego_usuario,
        "usuario_id": ronda.usuario_id,
        "categoria_id": ronda.categoria_id,
        "cantidad_minijuegos": ronda.cantidad_minijuegos,
        "correctas": ronda.correctas,
        "es_perfecta": ronda.correctas == ronda.cantidad_minijuegos,
    }


@router.post("/gamificacion/sincronizar/{usuario_id}")
def sincronizar_gamificacion(usuario_id: int, db: Session = Depends(get_db)):
    return crud.sincronizar_gamificacion_usuario(db, usuario_id)


@router.get("/logros-usuario/{usuario_id}")
def obtener_logros_usuario(usuario_id: int, db: Session = Depends(get_db)):
    return crud.consultar_logros_usuario(db, usuario_id)


@router.post("/perfil/equipamiento")
def equipar_perfil(data: schemas.EquipamientoPerfilCrear, db: Session = Depends(get_db)):
    return crud.equipar_marco_y_titulo(db, data)


@router.get("/marcos")
def obtener_marcos(usuario_id: int | None = None, db: Session = Depends(get_db)):
    marcos = db.query(models.Marco).order_by(models.Marco.orden.asc()).all()
    salida = []
    for marco in marcos:
        disponible = None
        if usuario_id is not None:
            disponible = crud.marco_disponible(db, usuario_id, marco)
        salida.append(
            {
                "id_marco": marco.id_marco,
                "id": marco.id_marco,
                "nombre": marco.nombre,
                "imagen_url": marco.imagen_url,
                "nivel_requerido": marco.nivel_requerido,
                "orden": marco.orden,
                "disponible": disponible,
            }
        )
    return salida


@router.get("/titulos")
def obtener_titulos(usuario_id: int | None = None, db: Session = Depends(get_db)):
    titulos = db.query(models.Titulo).order_by(models.Titulo.orden.asc()).all()
    salida = []
    for titulo in titulos:
        disponible = None
        if usuario_id is not None:
            disponible = crud.titulo_disponible(db, usuario_id, titulo)
        salida.append(
            {
                "id_titulo": titulo.id_titulo,
                "id": titulo.id_titulo,
                "nombre": titulo.nombre,
                "nivel_requerido": titulo.nivel_requerido,
                "orden": titulo.orden,
                "disponible": disponible,
            }
        )
    return salida
