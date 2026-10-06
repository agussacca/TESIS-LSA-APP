from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.desafios_interactivos import generar_ronda_desafios

router = APIRouter(prefix="/api/desafios-interactivos", tags=["Desafíos interactivos"])


@router.get("/categorias/{categoria_id}/ronda")
def obtener_ronda_desafios(
    categoria_id: int,
    cantidad: int = Query(default=5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    return generar_ronda_desafios(
        db=db,
        categoria_id=categoria_id,
        cantidad=cantidad,
    )