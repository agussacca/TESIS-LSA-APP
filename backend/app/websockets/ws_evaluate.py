from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.lsa_engine.evaluate_session import (
    EvaluateSession,
    EvaluateSessionConfig,
)
from app.lsa_engine.frame_decoder import (
    FrameDecodeError,
    decode_base64_jpeg_to_bgr,
)

router = APIRouter(tags=["evaluate-websocket"])


@router.websocket("/ws/evaluate/test")
async def evaluate_test_socket(
    websocket: WebSocket,
    target_label: str = "A",
    auto_stop_seconds: float = 2.5,
    stabilize_seconds: float = 1.0,
    start_consecutive_frames: int = 10,
    release_consecutive_frames: int = 8,
):
    await websocket.accept()

    session = None

    try:
        session = EvaluateSession(
            EvaluateSessionConfig(
                target_label=target_label,
                auto_stop_seconds=auto_stop_seconds,
                stabilize_seconds=stabilize_seconds,
                start_consecutive_frames=start_consecutive_frames,
                release_consecutive_frames=release_consecutive_frames,
            )
        )

        await websocket.send_json({
            "type": "evaluate_connection",
            "status": "connected",
            "target_label": session.target_label,
            "required_hands_target": session.required_hands_target,
            "message": "WebSocket de evaluación conectado.",
            "timestamp": datetime.utcnow().isoformat(),
        })

        counter = 0

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
                        "type": "evaluate_process_error",
                        "counter": counter,
                        "error": str(exc),
                        "message": "Ocurrió un error durante la evaluación.",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                continue

            if message_type == "reset_evaluate":
                if session is not None:
                    session.close()

                session = EvaluateSession(
                    EvaluateSessionConfig(
                        target_label=target_label,
                        auto_stop_seconds=auto_stop_seconds,
                        stabilize_seconds=stabilize_seconds,
                        start_consecutive_frames=start_consecutive_frames,
                        release_consecutive_frames=release_consecutive_frames,
                    )
                )

                await websocket.send_json({
                    "type": "evaluate_reset",
                    "target_label": session.target_label,
                    "message": "Sesión de evaluación reiniciada.",
                    "timestamp": datetime.utcnow().isoformat(),
                })

                continue

            await websocket.send_json({
                "type": "evaluate_message",
                "received_type": message_type,
                "message": "Mensaje recibido por WebSocket de evaluación.",
                "timestamp": datetime.utcnow().isoformat(),
            })

    except WebSocketDisconnect:
        if session is not None:
            session.close()

        print("Cliente WebSocket de evaluación desconectado")