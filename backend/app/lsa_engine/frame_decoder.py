from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

import cv2
import numpy as np


class FrameDecodeError(ValueError):
    """Error controlado al decodificar un frame enviado desde el frontend."""


@dataclass(frozen=True)
class DecodedFrame:
    image_bgr: np.ndarray
    width: int
    height: int
    channels: int
    size_bytes: int


def estimate_base64_size_bytes(base64_text: str) -> int:
    if not base64_text:
        return 0

    padding = base64_text.count("=")
    return int((len(base64_text) * 3) / 4) - padding


def normalize_base64_payload(image_base64: str) -> str:
    """
    Acepta tanto base64 puro como data URLs:
    - "/9j/4AAQSk..."
    - "data:image/jpeg;base64,/9j/4AAQSk..."
    """
    if not image_base64:
        raise FrameDecodeError("El frame llegó vacío.")

    if "," in image_base64 and image_base64.strip().startswith("data:"):
        return image_base64.split(",", 1)[1]

    return image_base64


def decode_base64_jpeg_to_bgr(
    image_base64: str,
    *,
    max_size_bytes: int = 2_500_000,
) -> DecodedFrame:
    normalized_base64 = normalize_base64_payload(image_base64)
    estimated_size = estimate_base64_size_bytes(normalized_base64)

    if estimated_size <= 0:
        raise FrameDecodeError("El tamaño estimado del frame es inválido.")

    if estimated_size > max_size_bytes:
        raise FrameDecodeError(
            f"El frame supera el tamaño máximo permitido "
            f"({estimated_size} bytes > {max_size_bytes} bytes)."
        )

    try:
        image_bytes = base64.b64decode(normalized_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameDecodeError("El contenido base64 del frame no es válido.") from exc

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)

    if np_buffer.size == 0:
        raise FrameDecodeError("El buffer NumPy del frame está vacío.")

    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FrameDecodeError("OpenCV no pudo decodificar el JPEG recibido.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise FrameDecodeError(
            f"Formato de imagen inesperado: shape={image_bgr.shape}."
        )

    height, width, channels = image_bgr.shape

    return DecodedFrame(
        image_bgr=image_bgr,
        width=int(width),
        height=int(height),
        channels=int(channels),
        size_bytes=len(image_bytes),
    )