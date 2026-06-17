#ws_spell.py
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.lsa_engine.frame_decoder import (
    FrameDecodeError,
    decode_base64_jpeg_to_bgr,
)
from app.lsa_engine.free_spell_session import (
    FreeSpellSession,
    FreeSpellSessionConfig,
)

router = APIRouter(tags=["spell-websocket"])


@router.websocket("/ws/spell/free")
async def spell_free_socket(
    websocket: WebSocket,
    record_seconds: float = 2.0,
    stabilize_seconds: float = 1.0,
    start_consecutive_frames: int = 10,
    release_consecutive_frames: int = 8,
    start_min_hands: int = 1,
    min_confidence: float = 0.70,
    min_margin: float = 0.15,
):
    await websocket.accept()

    session = FreeSpellSession(
        FreeSpellSessionConfig(
            record_seconds=record_seconds,
            stabilize_seconds=stabilize_seconds,
            start_consecutive_frames=start_consecutive_frames,
            release_consecutive_frames=release_consecutive_frames,
            start_min_hands=start_min_hands,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
    )

    await websocket.send_json({
        "type": "free_spell_connection",
        "status": "connected",
        "message": "WebSocket de deletreo libre conectado.",
        "timestamp": datetime.utcnow().isoformat(),
    })

    counter = 0

    try:
        while True:
            message = await websocket.receive_json()
            counter += 1

            message_type = message.get("type")

            if message_type == "camera_frame":
                try:
                    decoded = decode_base64_jpeg_to_bgr(
                        message.get("image_base64", "")
                    )

                    timestamp_ms = int(
                        message.get("captured_at")
                        or message.get("timestamp")
                        or counter
                    )

                    result = session.process_frame(
                        decoded.image_bgr,
                        timestamp_ms=timestamp_ms,
                        orientation=message.get("orientation"),
                        mirrored=message.get("mirrored"),
                    )

                    result.update({
                        "counter": counter,
                        "frontend_width": message.get("width"),
                        "frontend_height": message.get("height"),
                        "decoded_width": decoded.width,
                        "decoded_height": decoded.height,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                    await websocket.send_json(result)

                except FrameDecodeError as exc:
                    await websocket.send_json({
                        "type": "frame_decode_error",
                        "counter": counter,
                        "error": str(exc),
                        "message": "No se pudo decodificar el frame recibido.",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                except Exception as exc:
                    await websocket.send_json({
                        "type": "free_spell_process_error",
                        "counter": counter,
                        "error": str(exc),
                        "message": "Ocurrió un error durante el deletreo libre.",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                continue

            if message_type == "delete_last":
                session.delete_last_character()

                await websocket.send_json({
                    "type": "free_spell_text_update",
                    "spelled_text": session.spelled_text,
                    "message": "Último carácter eliminado.",
                    "timestamp": datetime.utcnow().isoformat(),
                })

                continue

            if message_type == "append_space":
                session.append_space()

                await websocket.send_json({
                    "type": "free_spell_text_update",
                    "spelled_text": session.spelled_text,
                    "message": "Espacio agregado.",
                    "timestamp": datetime.utcnow().isoformat(),
                })

                continue

            if message_type == "clear_text":
                session.clear_text()

                await websocket.send_json({
                    "type": "free_spell_text_update",
                    "spelled_text": session.spelled_text,
                    "message": "Texto limpiado.",
                    "timestamp": datetime.utcnow().isoformat(),
                })

                continue

            if message_type == "reset_state":
                session.reset_state_only()

                await websocket.send_json({
                    "type": "free_spell_reset",
                    "spelled_text": session.spelled_text,
                    "message": "Estado de deletreo reiniciado.",
                    "timestamp": datetime.utcnow().isoformat(),
                })

                continue

            await websocket.send_json({
                "type": "free_spell_message",
                "received_type": message_type,
                "spelled_text": session.spelled_text,
                "message": "Mensaje recibido por WebSocket de deletreo libre.",
                "timestamp": datetime.utcnow().isoformat(),
            })

    except WebSocketDisconnect:
        session.close()
        print("Cliente WebSocket de deletreo libre desconectado")