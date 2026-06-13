import { useState } from "react";
import CameraView from "../camera/CameraView";
import { useRecognitionSocket } from "../../hooks/useRecognitionSocket";

export default function CameraFrameDiagnostics() {
  const [socketEnabled, setSocketEnabled] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [sentFrames, setSentFrames] = useState(0);

  const {
    connected,
    lastMessage,
    error,
    sendCameraFrame,
  } = useRecognitionSocket(socketEnabled);

  function handleFrame(framePayload) {
    const sent = sendCameraFrame(framePayload);

    if (sent) {
      setSentFrames((value) => value + 1);
    }
  }

  return (
    <section className="camera-frame-diagnostics card">
      <div className="backend-diagnostics-header">
        <div>
          <h3>Diagnóstico cámara + WebSocket</h3>
          <p>Envía frames reales de la cámara al backend. La predicción sigue siendo simulada.</p>
        </div>

        <span className={connected ? "diag-status ok" : "diag-status error"}>
          {connected ? "WS OK" : "WS desconectado"}
        </span>
      </div>

      <div className="diag-actions">
        <button
          className="secondary"
          onClick={() => setSocketEnabled((value) => !value)}
        >
          {socketEnabled ? "Cerrar WebSocket" : "Abrir WebSocket"}
        </button>

        <button
          className="primary"
          onClick={() => setCaptureEnabled((value) => !value)}
          disabled={!connected}
        >
          {captureEnabled ? "Detener envío de frames" : "Enviar frames"}
        </button>
      </div>

      {error && <p className="diag-error">{error}</p>}

      <div className="camera-frame-grid">
        <CameraView
          title="Cámara real"
          subtitle="Activá la cámara y luego enviá frames al backend con la misma orientación usada en entrenamiento."
          captureEnabled={connected && captureEnabled}
          captureFps={2}
          jpegQuality={0.65}
          maxCaptureWidth={640}
          mirrorPreview={false}
          mirrorCapture={false}
          onFrame={handleFrame}
        />

        <div className="diag-box">
          <strong>Respuesta del backend</strong>

          <p>
            Frames enviados: <strong>{sentFrames}</strong>
          </p>

          {lastMessage ? (
            <pre>{JSON.stringify(lastMessage, null, 2)}</pre>
          ) : (
            <p className="diag-error">
              Todavía no hay respuesta de frames.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}