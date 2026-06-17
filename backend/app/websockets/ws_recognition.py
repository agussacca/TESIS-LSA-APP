#ws_recognition.py
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.lsa_engine.frame_decoder import (
    FrameDecodeError,
    decode_base64_jpeg_to_bgr,
)
from app.lsa_engine.frame_processor import (
    FrameMetadata,
    FrameProcessor,
)

router = APIRouter(tags=["recognition-websocket"])


@router.websocket("/ws/recognition/test")
async def recognition_test_socket(websocket: WebSocket):
    await websocket.accept()

    # Importante:
    # FrameProcessor debe ser por conexión/sesión.
    # Si fuera global, se mezclarían captured_items de distintos usuarios
    # o reconexiones.
    frame_processor = FrameProcessor()

    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "message": "WebSocket de reconocimiento conectado",
        "timestamp": datetime.utcnow().isoformat(),
    })

    counter = 0
    received_frames = 0
    decoded_frames = 0
    processed_frames = 0

    try:
        while True:
            message = await websocket.receive_json()
            counter += 1

            message_type = message.get("type")

            if message_type in {"reset_capture", "reset_buffer"}:
                frame_processor.reset_capture()

                await websocket.send_json({
                    "type": "capture_reset",
                    "message": "Captura reiniciada.",
                    "counter": counter,
                    "received_frames": received_frames,
                    "decoded_frames": decoded_frames,
                    "processed_frames": processed_frames,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                continue

            if message_type == "camera_frame":
                received_frames += 1
                image_base64 = message.get("image_base64", "")

                try:
                    decoded = decode_base64_jpeg_to_bgr(image_base64)
                    decoded_frames += 1

                    metadata = FrameMetadata(
                        width=decoded.width,
                        height=decoded.height,
                        channels=decoded.channels,
                        size_bytes=decoded.size_bytes,
                        mirrored=message.get("mirrored"),
                        orientation=message.get("orientation"),
                        timestamp_ms=int(
                            message.get("captured_at")
                            or message.get("timestamp")
                            or counter
                        ),
                    )

                    result = frame_processor.process_frame(
                        decoded.image_bgr,
                        metadata=metadata,
                        expected_hands=2,
                    )

                    processed_frames += 1

                    await websocket.send_json({
                        "type": "frame_processed",
                        "received_type": message_type,
                        "counter": counter,

                        "received_frames": received_frames,
                        "decoded_frames": decoded_frames,
                        "processed_frames": processed_frames,

                        "frontend_width": message.get("width"),
                        "frontend_height": message.get("height"),

                        "decoded_width": decoded.width,
                        "decoded_height": decoded.height,
                        "decoded_channels": decoded.channels,
                        "size_bytes": decoded.size_bytes,

                        "mirrored": message.get("mirrored"),
                        "orientation": message.get("orientation"),

                        "frame_ready": result.frame_ready,
                        "image_width": result.image_width,
                        "image_height": result.image_height,
                        "image_channels": result.image_channels,

                        "mean_b": result.mean_b,
                        "mean_g": result.mean_g,
                        "mean_r": result.mean_r,

                        "hands_detected": result.hands_detected,
                        "used_hands": result.used_hands,
                        "expected_hands": result.expected_hands,

                        "pose_detected": result.pose_detected,
                        "face_present": result.face_present,
                        "body_present": result.body_present,

                        "handedness_labels": result.handedness_labels,
                        "handedness_scores": result.handedness_scores,
                        "primary_label": result.primary_label,
                        "secondary_label": result.secondary_label,

                        "features_ready": result.features_ready,
                        "feature_vector_size": result.feature_vector_size,

                        "captured_items_count": result.captured_items_count,
                        "capture_duration_ms": result.capture_duration_ms,

                        "sequence_ready": result.sequence_ready,
                        "sequence_shape": result.sequence_shape,
                        "sampled_frames": result.sampled_frames,

                        "expected_hands_estimated": result.expected_hands_estimated,
                        "hand_ratio_any": result.hand_ratio_any,
                        "hand_ratio_expected": result.hand_ratio_expected,
                        "avg_detected_hands": result.avg_detected_hands,
                        "two_hand_ratio": result.two_hand_ratio,
                        "pose_ratio": result.pose_ratio,
                        "face_anchor_ratio": result.face_anchor_ratio,
                        "body_anchor_ratio": result.body_anchor_ratio,
                        "avg_anchor_scale": result.avg_anchor_scale,

                        "model_ready": result.model_ready,
                        "model_loaded": result.model_loaded,
                        "model_path": result.model_path,
                        "model_error": result.model_error,

                        "pred_label": result.pred_label,
                        "confidence": result.confidence,
                        "top_predictions": result.top_predictions,
                        "accepted": result.accepted,

                        "message": result.message,
                        "debug": result.debug,

                        "timestamp": datetime.utcnow().isoformat(),
                    })

                except FrameDecodeError as exc:
                    await websocket.send_json({
                        "type": "frame_decode_error",
                        "received_type": message_type,
                        "counter": counter,
                        "received_frames": received_frames,
                        "decoded_frames": decoded_frames,
                        "processed_frames": processed_frames,
                        "error": str(exc),
                        "message": "No se pudo decodificar el frame recibido.",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                except Exception as exc:
                    await websocket.send_json({
                        "type": "frame_process_error",
                        "received_type": message_type,
                        "counter": counter,
                        "received_frames": received_frames,
                        "decoded_frames": decoded_frames,
                        "processed_frames": processed_frames,
                        "error": str(exc),
                        "message": "Ocurrió un error al procesar el frame.",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                continue

            await websocket.send_json({
                "type": "dummy_prediction",
                "received_type": message_type,
                "counter": counter,
                "pred_label": "A",
                "confidence": 0.987,
                "accepted": counter % 3 == 0,
                "message": "Respuesta simulada del backend",
                "timestamp": datetime.utcnow().isoformat(),
            })

    except WebSocketDisconnect:
        frame_processor.close()
        print("Cliente WebSocket desconectado")