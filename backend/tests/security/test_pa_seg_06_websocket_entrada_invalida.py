from app.websockets import ws_evaluate


class _EvaluateSessionLiviana:
    """
    Reemplaza únicamente la inicialización pesada del mecanismo de reconocimiento.

    La ruta WebSocket, la recepción del mensaje, la decodificación del frame
    inválido y el manejo de FrameDecodeError siguen siendo los reales.
    """

    def __init__(self, config):
        self.target_label = config.target_label.strip().upper()
        self.required_hands_target = 1

    def close(self):
        pass


def test_pa_seg_06_websocket_entrada_invalida(
    app_cliente,
    monkeypatch,
):
    monkeypatch.setattr(
        ws_evaluate,
        "EvaluateSession",
        _EvaluateSessionLiviana,
    )

    with app_cliente.websocket_connect(
        "/ws/evaluate/test?target_label=A"
    ) as websocket:
        conexion = websocket.receive_json()

        assert conexion["type"] == "evaluate_connection"
        assert conexion["status"] == "connected"

        # Mensaje incompleto: camera_frame sin image_base64.
        websocket.send_json({
            "type": "camera_frame",
            "captured_at": 1,
        })

        error = websocket.receive_json()

        assert error["type"] == "frame_decode_error"
        assert "error" in error

        # Se envía otro mensaje para comprobar que la sesión continúa activa
        # después de la entrada inválida.
        websocket.send_json({
            "type": "mensaje_control",
        })

        respuesta_posterior = websocket.receive_json()

        assert respuesta_posterior["type"] == "evaluate_message"
        assert respuesta_posterior["received_type"] == "mensaje_control"
