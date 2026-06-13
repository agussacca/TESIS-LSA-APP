from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.core.config import settings
except Exception:  # pragma: no cover
    class _Settings:
        app_name = "SeñApp Backend"
        frontend_origin = "http://localhost:5173"
        api_prefix = "/api"

    settings = _Settings()

from app.db.database import SessionLocal, init_db
from app.db.seed import sembrar_datos_iniciales
from app.routers.resultados_practica import router as resultados_practica_router

app = FastAPI(title=settings.app_name)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    db = SessionLocal()
    try:
        sembrar_datos_iniciales(db)
    finally:
        db.close()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[getattr(settings, "frontend_origin", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from app.api.routes_health import router as health_router

    app.include_router(health_router, prefix=getattr(settings, "api_prefix", "/api"))
except Exception:
    pass

try:
    from app.websockets.ws_recognition import router as recognition_ws_router

    app.include_router(recognition_ws_router)
except Exception:
    pass

try:
    from app.websockets.ws_evaluate import router as evaluate_ws_router

    app.include_router(evaluate_ws_router)
except Exception:
    pass

try:
    from app.websockets.ws_spell import router as spell_ws_router

    app.include_router(spell_ws_router)
except Exception:
    pass

app.include_router(resultados_practica_router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SeñApp Backend",
    }
