from __future__ import annotations

import random
import re
import unicodedata
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db import models


PROVINCIA_CODIGOS: dict[str, str] = {
    "salta": "ARA",
    "santiago_del_estero": "ARG",
    "la_rioja": "ARF",
    "jujuy": "ARY",
    "tucuman": "ART",
    "catamarca": "ARK",
    "san_luis": "ARD",
    "cordoba": "ARX",
    "san_juan": "ARJ",
    "mendoza": "ARM",
}


def _slug(value: str) -> str:
    value = str(value or "").strip()
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.lower()
    value = value.replace("ñ", "n")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _serializar_senia(senia: models.Senia) -> dict[str, Any]:
    return {
        "id_senia": senia.id_senia,
        "id": senia.id_senia,
        "codigo": _slug(senia.nombre),
        "nombre": senia.nombre,
        "descripcion": senia.descripcion,
        "imagen_url": senia.imagen_url,
        "video_url": senia.video_url,
        "orden": senia.orden,
    }


def _obtener_categoria(db: Session, categoria_id: int) -> models.CategoriaAprendizaje:
    categoria = (
        db.query(models.CategoriaAprendizaje)
        .filter(models.CategoriaAprendizaje.id_categoria_aprendizaje == categoria_id)
        .first()
    )

    if categoria is None:
        raise HTTPException(status_code=404, detail="Categoría no encontrada.")

    return categoria


def _obtener_senias_categoria(db: Session, categoria_id: int) -> list[models.Senia]:
    return (
        db.query(models.Senia)
        .filter(models.Senia.categoria_id == categoria_id)
        .order_by(models.Senia.orden.asc())
        .all()
    )


def _mezclar(items: list[Any]) -> list[Any]:
    copia = list(items)
    random.shuffle(copia)
    return copia


def _elegir_senias(senias: list[models.Senia], cantidad: int) -> list[models.Senia]:
    if len(senias) < cantidad:
        raise HTTPException(
            status_code=400,
            detail=f"La categoría no tiene suficientes señas para generar este desafío. Se requieren {cantidad}.",
        )

    return random.sample(senias, cantidad)


def _opciones_multiple(
    senias: list[models.Senia],
    correcta: models.Senia,
    cantidad: int = 4,
) -> list[models.Senia]:
    distractores = [senia for senia in senias if senia.id_senia != correcta.id_senia]

    if len(distractores) < cantidad - 1:
        raise HTTPException(
            status_code=400,
            detail="La categoría no tiene suficientes señas para generar opciones de selección múltiple.",
        )

    opciones = random.sample(distractores, cantidad - 1)
    opciones.append(correcta)
    return _mezclar(opciones)


def _desafio_senia_a_nombre(
    senias: list[models.Senia],
    orden: int,
) -> dict[str, Any]:
    correcta = random.choice(senias)
    opciones = _opciones_multiple(senias, correcta, 4)

    return {
        "id": f"sign-to-word-{orden}",
        "tipo": "sign_to_word",
        "titulo": "¿Qué significa esta seña?",
        "orden": orden,
        "senia": _serializar_senia(correcta),
        "opciones": [
            {
                "id_senia": opcion.id_senia,
                "id": opcion.id_senia,
                "nombre": opcion.nombre,
            }
            for opcion in opciones
        ],
        "respuesta_correcta": {
            "id_senia": correcta.id_senia,
            "nombre": correcta.nombre,
        },
    }


def _desafio_nombre_a_senia(
    senias: list[models.Senia],
    orden: int,
) -> dict[str, Any]:
    correcta = random.choice(senias)
    opciones = _opciones_multiple(senias, correcta, 4)

    return {
        "id": f"word-to-sign-{orden}",
        "tipo": "word_to_sign",
        "titulo": f"Elegí la seña correspondiente a: {correcta.nombre}",
        "orden": orden,
        "nombre": correcta.nombre,
        "opciones": [_serializar_senia(opcion) for opcion in opciones],
        "respuesta_correcta": {
            "id_senia": correcta.id_senia,
            "nombre": correcta.nombre,
        },
    }


def _desafio_asociacion(
    senias: list[models.Senia],
    orden: int,
) -> dict[str, Any]:
    seleccionadas = _elegir_senias(senias, 4)
    nombres = _mezclar([senia.nombre for senia in seleccionadas])

    return {
        "id": f"association-{orden}",
        "tipo": "association",
        "titulo": "Arrastrá cada nombre a su tarjeta de seña",
        "orden": orden,
        "senias": [_serializar_senia(senia) for senia in seleccionadas],
        "nombres": nombres,
        "respuesta_correcta": {
            str(senia.id_senia): senia.nombre
            for senia in seleccionadas
        },
    }


def _desafio_ordenar_numeros(
    senias: list[models.Senia],
    orden: int,
) -> dict[str, Any]:
    seleccionadas = _elegir_senias(senias, 4)
    respuesta_correcta = sorted(seleccionadas, key=lambda senia: senia.orden)
    opciones = _mezclar(seleccionadas)

    return {
        "id": f"order-numbers-{orden}",
        "tipo": "order_numbers",
        "titulo": "Completá la secuencia",
        "orden": orden,
        "opciones": [_serializar_senia(senia) for senia in opciones],
        "respuesta_correcta": [senia.id_senia for senia in respuesta_correcta],
    }


def _desafio_mapa_provincias(
    senias: list[models.Senia],
    orden: int,
) -> dict[str, Any]:
    provincias_disponibles = [
        senia
        for senia in senias
        if _slug(senia.nombre) in PROVINCIA_CODIGOS
    ]

    seleccionadas = _elegir_senias(provincias_disponibles, 3)

    targets = []
    for senia in seleccionadas:
        codigo = _slug(senia.nombre)
        targets.append({
            **_serializar_senia(senia),
            "provincia_codigo": PROVINCIA_CODIGOS[codigo],
        })

    return {
        "id": f"province-map-{orden}",
        "tipo": "province_map",
        "titulo": "Ubicá las señas en el mapa",
        "orden": orden,
        "targets": targets,
        "opciones": _mezclar(targets),
        "respuesta_correcta": {
            str(target["id_senia"]): target["provincia_codigo"]
            for target in targets
        },
    }


def generar_ronda_desafios(
    db: Session,
    categoria_id: int,
    cantidad: int = 5,
) -> dict[str, Any]:
    categoria = _obtener_categoria(db, categoria_id)
    senias = _obtener_senias_categoria(db, categoria.id_categoria_aprendizaje)

    if len(senias) < 4:
        raise HTTPException(
            status_code=400,
            detail="La categoría no tiene suficientes señas para generar desafíos interactivos.",
        )

    categoria_slug = _slug(categoria.nombre)

    tipos_base = ["sign_to_word", "word_to_sign", "association"]
    tipos = list(tipos_base)

    if categoria_slug == "numeros":
        tipos.append("order_numbers")

    if categoria_slug == "provincias":
        tipos.append("province_map")

    while len(tipos) < cantidad:
        tipos.append(random.choice(tipos_base))

    tipos = tipos[:cantidad]

    desafios = []
    for index, tipo in enumerate(tipos, start=1):
        if tipo == "sign_to_word":
            desafios.append(_desafio_senia_a_nombre(senias, index))
        elif tipo == "word_to_sign":
            desafios.append(_desafio_nombre_a_senia(senias, index))
        elif tipo == "association":
            desafios.append(_desafio_asociacion(senias, index))
        elif tipo == "order_numbers":
            desafios.append(_desafio_ordenar_numeros(senias, index))
        elif tipo == "province_map":
            desafios.append(_desafio_mapa_provincias(senias, index))

    return {
        "categoria_id": categoria.id_categoria_aprendizaje,
        "categoria_nombre": categoria.nombre,
        "cantidad": len(desafios),
        "desafios": desafios,
    }