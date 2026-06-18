from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db

JWT_ALGORITHM = "HS256"
JWT_SECRET = os.getenv("SENAPP_JWT_SECRET", "senapp-dev-secret-change-me")
JWT_EXPIRE_MINUTES = int(os.getenv("SENAPP_JWT_EXPIRE_MINUTES", "10080"))  # 7 días
PBKDF2_ITERATIONS = 210_000
TOKEN_SCHEME = HTTPBearer(auto_error=False)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("La contraseña no puede estar vacía.")
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${_b64url_encode(derived)}"


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False

    # Compatibilidad con usuarios viejos creados antes de implementar hash.
    if not password_hash.startswith("pbkdf2_sha256$"):
        return hmac.compare_digest(password_hash, password)

    try:
        _, iterations_raw, salt, stored_hash = password_hash.split("$", 3)
        iterations = int(iterations_raw)
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return hmac.compare_digest(_b64url_encode(derived), stored_hash)
    except Exception:
        return False


def create_access_token(subject: str | int, extra_claims: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)

    header = {"typ": "JWT", "alg": JWT_ALGORITHM}
    signing_input = ".".join(
        [
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ]
    )
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_access_token(token: str) -> dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}"
        expected = hmac.new(JWT_SECRET.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
        received = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected, received):
            raise credentials_exception

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        exp = int(payload.get("exp", 0))
        if exp < int(datetime.now(timezone.utc).timestamp()):
            raise credentials_exception
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise credentials_exception from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(TOKEN_SCHEME),
    db: Session = Depends(get_db),
) -> models.Usuario:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No se enviaron credenciales de autenticación.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    subject = payload.get("sub")
    try:
        usuario_id = int(subject)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Token inválido.") from exc

    usuario = db.query(models.Usuario).filter(models.Usuario.id_usuario == usuario_id).first()
    if usuario is None:
        raise HTTPException(status_code=401, detail="Usuario no encontrado para el token enviado.")
    return usuario
