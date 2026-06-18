from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from app.db import models, schemas
from app.core.security import hash_password, verify_password
from app.services.estadisticas import (
    agrupar_progreso_por_letra,
    calcular_resumen_estadisticas,
    intento_aceptado,
    ronda_perfecta,
    validar_ronda_minijuego,
)
from app.services.gamificacion import (
    actualizar_racha_por_objetivo_diario,
    calcular_nivel,
    calcular_xp_ronda_minijuego,
    objetivo_puede_otorgar_xp,
    progreso_nivel,
)


OBJETIVO_REGLAS: dict[str, tuple[str, int]] = {
    "APRENDÉ 2 SEÑAS": ("senias_aprendidas", 2),
    "COMPLETÁ 3 MINIJUEGOS": ("minijuegos_correctos", 3),
    "PRACTICÁ CON CÁMARA": ("practicas_camara", 1),
    "APRENDÉ 8 SEÑAS NUEVAS": ("senias_aprendidas", 8),
    "COMPLETÁ 10 RONDAS": ("rondas_completadas", 10),
    "LOGRÁ 3 RONDAS PERFECTAS": ("rondas_perfectas", 3),
    "DELETREAR UNA PALABRA": ("palabras_deletreadas", 1),
    "COMPLETAR UNA RONDA": ("rondas_completadas", 1),
    "PRACTICAR CON CÁMARA": ("practicas_camara", 1),
    "PRACTICAR DURANTE LA SEMANA": ("practicas_camara", 1),
}


LOGRO_METADATA: dict[str, dict[str, int | str]] = {
    'Primer compromiso': {"familia": 'Objetivos', "orden": 1},
    'Rutina en marcha': {"familia": 'Objetivos', "orden": 2},
    'Hábito firme': {"familia": 'Objetivos', "orden": 3},
    'Meta semanal cumplida': {"familia": 'Objetivos', "orden": 4},
    'Semana productiva': {"familia": 'Objetivos', "orden": 5},
    'Primer día activo': {"familia": 'Rachas', "orden": 10},
    'Tres días en marcha': {"familia": 'Rachas', "orden": 11},
    'Semana imparable': {"familia": 'Rachas', "orden": 12},
    'Quince días de constancia': {"familia": 'Rachas', "orden": 13},
    'Mes perfecto de práctica': {"familia": 'Rachas', "orden": 14},
    'Primer ascenso': {"familia": 'Niveles', "orden": 20},
    'Aprendiz destacado': {"familia": 'Niveles', "orden": 21},
    'Dominio en crecimiento': {"familia": 'Niveles', "orden": 22},
    'Abecedario Conquistado': {"familia": 'Abecedario', "orden": 30},
    'Maestría Alfabética': {"familia": 'Abecedario', "orden": 31},
    'Paleta Aprendida': {"familia": 'Colores', "orden": 40},
    'Maestría Cromática': {"familia": 'Colores', "orden": 41},
    'Primeras Conversaciones': {"familia": 'Comunicación Básica', "orden": 50},
    'Diálogo Impecable': {"familia": 'Comunicación Básica', "orden": 51},
    'Espíritu Deportivo': {"familia": 'Deportes', "orden": 60},
    'Jugada Perfecta': {"familia": 'Deportes', "orden": 61},
    'Lazos Aprendidos': {"familia": 'Familia', "orden": 70},
    'Lazos Firmes': {"familia": 'Familia', "orden": 71},
    'Números Dominados': {"familia": 'Números', "orden": 80},
    'Cuenta Clara': {"familia": 'Números', "orden": 81},
    'Conocedor de Territorios': {"familia": 'Provincias', "orden": 90},
    'Como la Palma de Mi Mano': {"familia": 'Provincias', "orden": 91},
}


def _metadata_logro(nombre: str) -> dict[str, int | str]:
    return LOGRO_METADATA.get(str(nombre), {"familia": "General", "orden": 999})


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _clave_periodo(periodicidad: str, fecha: date | None = None) -> str:
    fecha = fecha or date.today()
    periodicidad_normalizada = str(periodicidad or "").upper()
    if periodicidad_normalizada == "SEMANAL":
        iso = fecha.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return fecha.isoformat()


def obtener_usuario(db: Session, usuario_id: int) -> models.Usuario | None:
    return db.query(models.Usuario).filter(models.Usuario.id_usuario == usuario_id).first()


def obtener_usuario_o_404(db: Session, usuario_id: int) -> models.Usuario:
    usuario = obtener_usuario(db, usuario_id)
    if usuario is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    return usuario


def obtener_usuario_por_email(db: Session, email: str) -> models.Usuario | None:
    email_normalizado = str(email or "").strip().lower()
    if not email_normalizado:
        return None
    return db.query(models.Usuario).filter(models.Usuario.email == email_normalizado).first()


def crear_usuario_registrado(db: Session, data: schemas.UsuarioCrear) -> models.Usuario:
    usuario_existente = obtener_usuario_por_email(db, data.email)
    if usuario_existente is not None:
        raise HTTPException(status_code=400, detail="Ya existe un usuario registrado con ese correo electrónico.")

    usuario = models.Usuario(
        email=data.email,
        password_hash=hash_password(data.password),
        nombre_visible=data.nombre_visible,
        foto_perfil_url=data.foto_perfil_url,
    )
    db.add(usuario)
    db.flush()

    progreso = models.ProgresoUsuario(usuario_id=usuario.id_usuario)
    db.add(progreso)
    db.commit()
    db.refresh(usuario)
    return usuario


def autenticar_usuario(db: Session, email: str, password: str) -> models.Usuario | None:
    usuario = obtener_usuario_por_email(db, email)
    if usuario is None:
        return None
    if not verify_password(password, usuario.password_hash):
        return None

    # Si el usuario venía de una versión anterior con contraseña plana, migrar a hash.
    if not str(usuario.password_hash or "").startswith("pbkdf2_sha256$"):
        usuario.password_hash = hash_password(password)
        db.add(usuario)
        db.commit()
        db.refresh(usuario)
    return usuario


def crear_usuario(db: Session, data: schemas.UsuarioCrear) -> models.Usuario:
    usuario_existente = obtener_usuario_por_email(db, data.email)
    if usuario_existente is not None:
        return usuario_existente

    usuario = models.Usuario(
        email=data.email,
        password_hash=hash_password(data.password),
        nombre_visible=data.nombre_visible,
        foto_perfil_url=data.foto_perfil_url,
    )
    db.add(usuario)
    db.flush()

    progreso = models.ProgresoUsuario(usuario_id=usuario.id_usuario)
    db.add(progreso)
    db.commit()
    db.refresh(usuario)
    return usuario


def obtener_o_crear_progreso(db: Session, usuario_id: int) -> models.ProgresoUsuario:
    obtener_usuario_o_404(db, usuario_id)
    progreso = db.query(models.ProgresoUsuario).filter(models.ProgresoUsuario.usuario_id == usuario_id).first()
    if progreso is None:
        progreso = models.ProgresoUsuario(usuario_id=usuario_id)
        db.add(progreso)
        db.commit()
        db.refresh(progreso)
    return progreso


def listar_contenido_aprendizaje(db: Session) -> list[models.CategoriaAprendizaje]:
    return (
        db.query(models.CategoriaAprendizaje)
        .options(joinedload(models.CategoriaAprendizaje.senias))
        .order_by(models.CategoriaAprendizaje.orden.asc())
        .all()
    )


def crear_intento_practica(db: Session, data: schemas.IntentoPracticaCrear) -> models.IntentoPractica:
    obtener_usuario_o_404(db, data.usuario_id)
    intento = models.IntentoPractica(
        usuario_id=data.usuario_id,
        letra_esperada=data.letra_esperada,
        letra_predicha=data.letra_predicha,
        validado=data.validado,
    )
    db.add(intento)
    db.commit()
    db.refresh(intento)
    return intento


def listar_intentos_usuario(db: Session, usuario_id: int) -> list[models.IntentoPractica]:
    obtener_usuario_o_404(db, usuario_id)
    return db.query(models.IntentoPractica).filter(models.IntentoPractica.usuario_id == usuario_id).all()


def crear_palabra_deletreada(db: Session, data: schemas.PalabraDeletreadaCrear) -> models.PalabraDeletreadaUsuario:
    obtener_usuario_o_404(db, data.usuario_id)
    palabra = models.PalabraDeletreadaUsuario(usuario_id=data.usuario_id, palabra=data.palabra)
    db.add(palabra)
    db.commit()
    db.refresh(palabra)
    return palabra


def listar_palabras_usuario(db: Session, usuario_id: int) -> list[models.PalabraDeletreadaUsuario]:
    obtener_usuario_o_404(db, usuario_id)
    return db.query(models.PalabraDeletreadaUsuario).filter(models.PalabraDeletreadaUsuario.usuario_id == usuario_id).all()


def crear_ronda_minijuego(db: Session, data: schemas.RondaMinijuegoCrear) -> models.RondaMinijuegoUsuario:
    obtener_usuario_o_404(db, data.usuario_id)
    categoria = (
        db.query(models.CategoriaAprendizaje)
        .filter(models.CategoriaAprendizaje.id_categoria_aprendizaje == data.categoria_id)
        .first()
    )
    if categoria is None:
        raise HTTPException(status_code=404, detail="Categoría no encontrada.")

    try:
        validar_ronda_minijuego(data.cantidad_minijuegos, data.correctas)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ronda = models.RondaMinijuegoUsuario(
        usuario_id=data.usuario_id,
        categoria_id=data.categoria_id,
        cantidad_minijuegos=data.cantidad_minijuegos,
        correctas=data.correctas,
    )
    db.add(ronda)
    db.commit()
    db.refresh(ronda)
    return ronda


def listar_rondas_usuario(db: Session, usuario_id: int) -> list[models.RondaMinijuegoUsuario]:
    obtener_usuario_o_404(db, usuario_id)
    return (
        db.query(models.RondaMinijuegoUsuario)
        .options(joinedload(models.RondaMinijuegoUsuario.categoria))
        .filter(models.RondaMinijuegoUsuario.usuario_id == usuario_id)
        .all()
    )


def obtener_progreso_por_letra(db: Session, usuario_id: int) -> list[dict[str, Any]]:
    intentos = listar_intentos_usuario(db, usuario_id)
    return agrupar_progreso_por_letra(intentos)


def _serializar_usuario_basico(usuario: models.Usuario) -> dict[str, Any]:
    marco = usuario.marco_equipado
    titulo = usuario.titulo_equipado
    return {
        "id_usuario": usuario.id_usuario,
        "email": usuario.email,
        "nombre_visible": usuario.nombre_visible,
        "foto_perfil_url": usuario.foto_perfil_url,
        "marco_equipado_id": usuario.marco_equipado_id,
        "titulo_equipado_id": usuario.titulo_equipado_id,
        "marco_equipado": (
            {
                "id_marco": marco.id_marco,
                "id": marco.id_marco,
                "nombre": marco.nombre,
                "imagen_url": marco.imagen_url,
                "nivel_requerido": marco.nivel_requerido,
                "orden": marco.orden,
                "disponible": True,
            }
            if marco is not None
            else None
        ),
        "titulo_equipado": (
            {
                "id_titulo": titulo.id_titulo,
                "id": titulo.id_titulo,
                "nombre": titulo.nombre,
                "nivel_requerido": titulo.nivel_requerido,
                "orden": titulo.orden,
                "disponible": True,
            }
            if titulo is not None
            else None
        ),
    }


def consultar_usuario_registrado(db: Session, usuario_id: int) -> dict[str, Any]:
    usuario = obtener_usuario_o_404(db, usuario_id)
    return _serializar_usuario_basico(usuario)


def actualizar_perfil_usuario(
    db: Session,
    usuario_id: int,
    data: schemas.UsuarioPerfilActualizar,
) -> dict[str, Any]:
    usuario = obtener_usuario_o_404(db, usuario_id)

    if data.nombre_visible is not None:
        usuario.nombre_visible = data.nombre_visible

    if data.foto_perfil_url is not None:
        usuario.foto_perfil_url = data.foto_perfil_url

    if data.email is not None:
        email_normalizado = data.email.strip().lower()
        usuario_existente = obtener_usuario_por_email(db, email_normalizado)
        if usuario_existente is not None and usuario_existente.id_usuario != usuario.id_usuario:
            raise HTTPException(status_code=400, detail="Ya existe un usuario registrado con ese correo electrónico.")
        usuario.email = email_normalizado

    if data.password is not None:
        usuario.password_hash = hash_password(data.password)

    db.add(usuario)
    db.commit()
    db.refresh(usuario)
    return _serializar_usuario_basico(usuario)


def obtener_panel_usuario(db: Session, usuario_id: int) -> dict[str, Any]:
    usuario = obtener_usuario_o_404(db, usuario_id)
    intentos = listar_intentos_usuario(db, usuario_id)
    palabras = listar_palabras_usuario(db, usuario_id)
    rondas = listar_rondas_usuario(db, usuario_id)
    progreso = obtener_o_crear_progreso(db, usuario_id)

    resumen = calcular_resumen_estadisticas(
        intentos=intentos,
        palabras_deletreadas=palabras,
        rondas=rondas,
    )
    progreso_nivel_data = progreso_nivel(int(progreso.xp_total))
    resumen["progreso"] = {
        "xp_total": int(progreso.xp_total),
        "nivel": int(progreso.nivel),
        "xp_nivel_actual": progreso_nivel_data["xp_nivel_actual"],
        "xp_siguiente_nivel": progreso_nivel_data["xp_siguiente_nivel"],
        "racha_actual": int(progreso.racha_actual),
        "racha_maxima": int(progreso.racha_maxima),
        "fecha_ultima_racha": progreso.fecha_ultima_racha.isoformat() if progreso.fecha_ultima_racha else None,
    }
    resumen["usuario"] = _serializar_usuario_basico(usuario)
    return resumen


def _contar_actividad(db: Session, usuario_id: int) -> dict[str, int]:
    intentos = listar_intentos_usuario(db, usuario_id)
    palabras = listar_palabras_usuario(db, usuario_id)
    rondas = listar_rondas_usuario(db, usuario_id)

    aceptados = [intento for intento in intentos if intento_aceptado(intento)]
    letras_aprendidas = {str(intento.letra_esperada).upper() for intento in aceptados}

    return {
        "practicas_camara": len(intentos),
        "intentos_aceptados": len(aceptados),
        "senias_aprendidas": len(letras_aprendidas),
        "palabras_deletreadas": len(palabras),
        "rondas_completadas": len(rondas),
        "rondas_perfectas": sum(1 for ronda in rondas if ronda_perfecta(ronda)),
        "minijuegos_correctos": sum(int(ronda.correctas) for ronda in rondas),
    }


def _regla_objetivo(objetivo: models.Objetivo) -> tuple[str, int]:
    return OBJETIVO_REGLAS.get(str(objetivo.nombre).strip().upper(), ("rondas_completadas", 1))


def _serializar_progreso(progreso: models.ProgresoUsuario) -> dict[str, Any]:
    extra = progreso_nivel(int(progreso.xp_total))
    return {
        "xp_total": int(progreso.xp_total),
        "nivel": int(progreso.nivel),
        "xp_nivel_actual": extra["xp_nivel_actual"],
        "xp_siguiente_nivel": extra["xp_siguiente_nivel"],
        "racha_actual": int(progreso.racha_actual),
        "racha_maxima": int(progreso.racha_maxima),
        "fecha_ultima_racha": progreso.fecha_ultima_racha.isoformat() if progreso.fecha_ultima_racha else None,
    }


def consultar_objetivos_usuario(db: Session, usuario_id: int) -> dict[str, Any]:
    obtener_usuario_o_404(db, usuario_id)
    progreso = obtener_o_crear_progreso(db, usuario_id)
    actividad = _contar_actividad(db, usuario_id)
    completados = db.query(models.ObjetivoCompletadoUsuario).filter(
        models.ObjetivoCompletadoUsuario.usuario_id == usuario_id
    ).all()

    completados_por_objetivo_periodo = {
        (item.objetivo_id, item.clave_periodo)
        for item in completados
    }

    diarios: list[dict[str, Any]] = []
    semanales: list[dict[str, Any]] = []

    objetivos = db.query(models.Objetivo).order_by(models.Objetivo.id_objetivo.asc()).all()
    for objetivo in objetivos:
        accion, requerido = _regla_objetivo(objetivo)
        actual = int(actividad.get(accion, 0))
        clave = _clave_periodo(str(objetivo.periodicidad))
        completado = actual >= requerido
        xp_otorgado = (objetivo.id_objetivo, clave) in completados_por_objetivo_periodo

        item = {
            "id": objetivo.id_objetivo,
            "codigo": _slug(objetivo.nombre),
            "titulo": objetivo.nombre,
            "descripcion": objetivo.descripcion,
            "actual": min(actual, requerido),
            "objetivo": requerido,
            "xp": objetivo.xp_recompensa,
            "completado": completado,
            "xp_otorgado": xp_otorgado,
        }

        if str(objetivo.periodicidad).upper() == "SEMANAL":
            semanales.append(item)
        else:
            diarios.append(item)

    return {
        "diarios": diarios,
        "semanales": semanales,
        "progreso": _serializar_progreso(progreso),
        "eventos": [],
    }


def sincronizar_gamificacion_usuario(db: Session, usuario_id: int) -> dict[str, Any]:
    obtener_usuario_o_404(db, usuario_id)
    progreso = obtener_o_crear_progreso(db, usuario_id)
    nivel_anterior = int(progreso.nivel)
    actividad = _contar_actividad(db, usuario_id)

    objetivos = db.query(models.Objetivo).order_by(models.Objetivo.id_objetivo.asc()).all()
    completados = db.query(models.ObjetivoCompletadoUsuario).filter(
        models.ObjetivoCompletadoUsuario.usuario_id == usuario_id
    ).all()

    nuevos_objetivos: list[dict[str, Any]] = []
    eventos: list[dict[str, Any]] = []
    xp_otorgada = 0
    completo_objetivo_diario = False

    for objetivo in objetivos:
        accion, requerido = _regla_objetivo(objetivo)
        actual = int(actividad.get(accion, 0))
        clave = _clave_periodo(str(objetivo.periodicidad))
        if actual < requerido:
            continue
        if not objetivo_puede_otorgar_xp(completados=completados, objetivo_id=objetivo.id_objetivo, clave_periodo=clave):
            continue

        completado = models.ObjetivoCompletadoUsuario(
            usuario_id=usuario_id,
            objetivo_id=objetivo.id_objetivo,
            clave_periodo=clave,
        )
        db.add(completado)
        db.flush()
        completados.append(completado)

        xp_otorgada += int(objetivo.xp_recompensa)
        if str(objetivo.periodicidad).upper() == "DIARIA":
            completo_objetivo_diario = True

        item = {
            "id_objetivo_completado_usuario": completado.id_objetivo_completado_usuario,
            "objetivo_id": objetivo.id_objetivo,
            "nombre": objetivo.nombre,
            "xp": objetivo.xp_recompensa,
            "clave_periodo": clave,
        }
        nuevos_objetivos.append(item)
        eventos.append({"tipo": "objetivo", "objetivo_id": objetivo.id_objetivo, "objetivo_titulo": objetivo.nombre, "mensaje": objetivo.descripcion, "xp": objetivo.xp_recompensa, "tipo_periodo": str(objetivo.periodicidad).lower()})

    xp_actividad = actividad["intentos_aceptados"] * 5
    xp_actividad += actividad["palabras_deletreadas"] * 10
    for ronda in listar_rondas_usuario(db, usuario_id):
        xp_actividad += calcular_xp_ronda_minijuego(
            cantidad_minijuegos=int(ronda.cantidad_minijuegos),
            correctas=int(ronda.correctas),
        )

    progreso.xp_total = int(xp_actividad + sum(obj.objetivo.xp_recompensa for obj in completados if obj.objetivo is not None))
    progreso.nivel = calcular_nivel(int(progreso.xp_total))

    if completo_objetivo_diario:
        racha = actualizar_racha_por_objetivo_diario(
            fecha_ultima_racha=progreso.fecha_ultima_racha,
            racha_actual=int(progreso.racha_actual),
            racha_maxima=int(progreso.racha_maxima),
            fecha_actual=date.today(),
        )
        progreso.racha_actual = int(racha["racha_actual"])
        progreso.racha_maxima = int(racha["racha_maxima"])
        progreso.fecha_ultima_racha = racha["fecha_ultima_racha"]

    if int(progreso.nivel) > nivel_anterior:
        eventos.append({
            "tipo": "nivel",
            "nivel_anterior": nivel_anterior,
            "nivel_nuevo": int(progreso.nivel),
            "mensaje": "Subiste de nivel.",
        })

    nuevos_logros = _desbloquear_logros(db, usuario_id, actividad, progreso)
    for logro in nuevos_logros:
        eventos.append({
            "tipo": "logro",
            "logro_id": logro["id_logro"],
            "logro_nombre": logro["nombre"],
            "logro_descripcion": logro["descripcion"],
            "logro_imagen_url": logro.get("imagen_url"),
            "logro_familia": logro.get("familia"),
            "mensaje": "Nuevo logro desbloqueado.",
        })

    db.commit()
    db.refresh(progreso)

    return {
        "progreso": _serializar_progreso(progreso),
        "objetivos_completados": nuevos_objetivos,
        "logros_desbloqueados": nuevos_logros,
        "eventos": eventos,
    }




def _contar_objetivos_completados(db: Session, usuario_id: int) -> dict[str, int]:
    completados = (
        db.query(models.ObjetivoCompletadoUsuario)
        .join(models.Objetivo, models.Objetivo.id_objetivo == models.ObjetivoCompletadoUsuario.objetivo_id)
        .filter(models.ObjetivoCompletadoUsuario.usuario_id == usuario_id)
        .all()
    )
    diarios = 0
    semanales = 0
    for item in completados:
        periodicidad = str(item.objetivo.periodicidad if item.objetivo is not None else "").upper()
        if periodicidad == "SEMANAL":
            semanales += 1
        else:
            diarios += 1
    return {"diarios": diarios, "semanales": semanales}


def _contar_rondas_perfectas_por_categoria(db: Session, usuario_id: int) -> dict[str, int]:
    conteo: dict[str, int] = {}
    for ronda in listar_rondas_usuario(db, usuario_id):
        nombre_categoria = str(ronda.categoria.nombre if ronda.categoria is not None else "")
        if not ronda_perfecta(ronda):
            continue
        conteo[nombre_categoria.lower()] = conteo.get(nombre_categoria.lower(), 0) + 1
    return conteo

def _condicion_logro(
    db: Session,
    usuario_id: int,
    nombre: str,
    descripcion: str,
    actividad: dict[str, int],
    progreso: models.ProgresoUsuario,
) -> bool:
    texto = f"{nombre} {descripcion}".lower()
    objetivos = _contar_objetivos_completados(db, usuario_id)
    perfectas_por_categoria = _contar_rondas_perfectas_por_categoria(db, usuario_id)

    if "primer objetivo diario" in texto:
        return objetivos["diarios"] >= 1
    if "10 objetivos diarios" in texto:
        return objetivos["diarios"] >= 10
    if "50 objetivos diarios" in texto:
        return objetivos["diarios"] >= 50
    if "primer objetivo semanal" in texto:
        return objetivos["semanales"] >= 1
    if "10 objetivos semanales" in texto:
        return objetivos["semanales"] >= 10

    if "racha de 30" in texto:
        return int(progreso.racha_maxima) >= 30
    if "racha de 15" in texto:
        return int(progreso.racha_maxima) >= 15
    if "racha de 7" in texto:
        return int(progreso.racha_maxima) >= 7
    if "racha de 3" in texto:
        return int(progreso.racha_maxima) >= 3
    if "objetivo diario en un día" in texto:
        return int(progreso.racha_maxima) >= 1 or objetivos["diarios"] >= 1

    if "nivel 10" in texto:
        return int(progreso.nivel) >= 10
    if "nivel 5" in texto:
        return int(progreso.nivel) >= 5
    if "nivel 2" in texto or "ascenso" in texto:
        return int(progreso.nivel) >= 2

    if "abecedario" in texto and "sin cometer ningún error" in texto:
        return perfectas_por_categoria.get("abecedario", 0) >= 5
    if "colores" in texto and "sin cometer ningún error" in texto:
        return perfectas_por_categoria.get("colores", 0) >= 5
    if "comunicación básica" in texto and "sin cometer ningún error" in texto:
        return perfectas_por_categoria.get("comunicación básica", 0) >= 5
    if "deportes" in texto and "sin cometer ningún error" in texto:
        return perfectas_por_categoria.get("deportes", 0) >= 5
    if "familia" in texto and "sin cometer ningún error" in texto:
        return perfectas_por_categoria.get("familia", 0) >= 5
    if "números" in texto and "sin cometer ningún error" in texto:
        return perfectas_por_categoria.get("números", 0) >= 5
    if "provincias" in texto and "sin cometer ningún error" in texto:
        return perfectas_por_categoria.get("provincias", 0) >= 5

    if "abecedario" in texto:
        return actividad["senias_aprendidas"] >= 27
    if "todas las señas" in texto:
        return False

    if "primera práctica" in texto or "primer paso" in texto:
        return actividad["intentos_aceptados"] >= 1
    if "ronda" in texto or "minijuego" in texto or "jugador" in texto:
        return actividad["rondas_completadas"] >= 1
    if "palabra" in texto or "delet" in texto:
        return actividad["palabras_deletreadas"] >= 1

    return actividad["practicas_camara"] >= 1

def _desbloquear_logros(db: Session, usuario_id: int, actividad: dict[str, int], progreso: models.ProgresoUsuario) -> list[dict[str, Any]]:
    existentes = {
        item.logro_id
        for item in db.query(models.LogroDesbloqueadoUsuario)
        .filter(models.LogroDesbloqueadoUsuario.usuario_id == usuario_id)
        .all()
    }
    nuevos: list[dict[str, Any]] = []

    for logro in db.query(models.Logro).order_by(models.Logro.id_logro.asc()).all():
        if logro.id_logro in existentes:
            continue
        if not _condicion_logro(db, usuario_id, logro.nombre, logro.descripcion, actividad, progreso):
            continue
        desbloqueo = models.LogroDesbloqueadoUsuario(usuario_id=usuario_id, logro_id=logro.id_logro)
        db.add(desbloqueo)
        db.flush()
        metadata = _metadata_logro(logro.nombre)
        nuevos.append({
            "id_logro": logro.id_logro,
            "logro_id": logro.id_logro,
            "nombre": logro.nombre,
            "descripcion": logro.descripcion,
            "imagen_url": logro.imagen_url,
            "familia": metadata["familia"],
            "orden": metadata["orden"],
        })
    return nuevos


def consultar_logros_usuario(db: Session, usuario_id: int) -> dict[str, Any]:
    obtener_usuario_o_404(db, usuario_id)
    desbloqueos = {
        item.logro_id: item
        for item in db.query(models.LogroDesbloqueadoUsuario)
        .filter(models.LogroDesbloqueadoUsuario.usuario_id == usuario_id)
        .all()
    }
    logros = db.query(models.Logro).order_by(models.Logro.id_logro.asc()).all()

    salida = []
    for logro in logros:
        desbloqueo = desbloqueos.get(logro.id_logro)
        metadata = _metadata_logro(logro.nombre)
        salida.append({
            "id_logro": logro.id_logro,
            "id": logro.id_logro,
            "codigo": _slug(logro.nombre),
            "familia": metadata["familia"],
            "nombre": logro.nombre,
            "descripcion": logro.descripcion,
            "imagen_url": logro.imagen_url,
            "orden": metadata["orden"],
            "desbloqueado": desbloqueo is not None,
            "fecha_desbloqueo": desbloqueo.fecha_desbloqueo if desbloqueo is not None else None,
        })

    salida.sort(key=lambda item: (str(item["familia"]), int(item["orden"])))
    cantidad_desbloqueados = sum(1 for item in salida if item["desbloqueado"])
    return {
        "total": len(salida),
        "desbloqueados": cantidad_desbloqueados,
        "pendientes": len(salida) - cantidad_desbloqueados,
        "logros": salida,
    }


def _cantidad_rondas_perfectas_deportes(db: Session, usuario_id: int) -> int:
    rondas = listar_rondas_usuario(db, usuario_id)
    total = 0
    for ronda in rondas:
        nombre_categoria = str(ronda.categoria.nombre if ronda.categoria is not None else "").lower()
        if nombre_categoria == "deportes" and ronda_perfecta(ronda):
            total += 1
    return total


def marco_disponible(db: Session, usuario_id: int, marco: models.Marco) -> bool:
    progreso = obtener_o_crear_progreso(db, usuario_id)
    if marco.nivel_requerido is not None:
        return int(progreso.nivel) >= int(marco.nivel_requerido)

    nombre = marco.nombre.strip().lower()
    perfectas_deportes = _cantidad_rondas_perfectas_deportes(db, usuario_id)
    if nombre == "fútbol" or nombre == "futbol":
        return perfectas_deportes >= 1
    if nombre == "tenis":
        return perfectas_deportes >= 2
    return False


def titulo_disponible(db: Session, usuario_id: int, titulo: models.Titulo) -> bool:
    progreso = obtener_o_crear_progreso(db, usuario_id)
    return int(progreso.nivel) >= int(titulo.nivel_requerido)


def equipar_marco_y_titulo(db: Session, data: schemas.EquipamientoPerfilCrear) -> dict[str, Any]:
    usuario = obtener_usuario_o_404(db, data.usuario_id)
    marco = db.query(models.Marco).filter(models.Marco.id_marco == data.marco_id).first()
    titulo = db.query(models.Titulo).filter(models.Titulo.id_titulo == data.titulo_id).first()

    if marco is None:
        raise HTTPException(status_code=404, detail="Marco no encontrado.")
    if titulo is None:
        raise HTTPException(status_code=404, detail="Título no encontrado.")
    if not marco_disponible(db, data.usuario_id, marco):
        raise HTTPException(status_code=403, detail="El marco no está disponible para este usuario.")
    if not titulo_disponible(db, data.usuario_id, titulo):
        raise HTTPException(status_code=403, detail="El título no está disponible para este usuario.")

    usuario.marco_equipado_id = marco.id_marco
    usuario.titulo_equipado_id = titulo.id_titulo
    db.commit()
    db.refresh(usuario)

    return {
        "usuario_id": usuario.id_usuario,
        "marco_equipado": {
            "id_marco": marco.id_marco,
            "nombre": marco.nombre,
            "imagen_url": marco.imagen_url,
        },
        "titulo_equipado": {
            "id_titulo": titulo.id_titulo,
            "nombre": titulo.nombre,
        },
    }


# Alias utilizado por versiones anteriores de main.py.
def sembrar_objetivos_iniciales(db: Session) -> None:
    from app.db.seed import sembrar_datos_iniciales

    sembrar_datos_iniciales(db)
