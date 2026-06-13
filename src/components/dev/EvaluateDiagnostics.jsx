import { useCallback, useEffect, useRef, useState } from "react";
import CameraView from "../camera/CameraView";
import { useEvaluateSocket } from "../../hooks/useEvaluateSocket";

const MAX_PENDING_FRAMES = 80;

export default function EvaluateDiagnostics() {
  const [targetLabel, setTargetLabel] = useState("A");
  const [socketEnabled, setSocketEnabled] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [finalResult, setFinalResult] = useState(null);
  const [sentFrames, setSentFrames] = useState(0);

  // Mantener 2.0 para respetar el protocolo usado en evaluate offline.
  const [autoStopSeconds, setAutoStopSeconds] = useState(2.0);

  const pendingFramesRef = useRef(0);
  const [pendingFrames, setPendingFrames] = useState(0);

  const {
    connected,
    lastMessage,
    error,
    sendCameraFrame,
    resetEvaluate,
  } = useEvaluateSocket({
    enabled: socketEnabled,
    targetLabel,
    autoStopSeconds,
  });

  useEffect(() => {
    if (!lastMessage) return;

    if (
      lastMessage.type === "evaluate_update" ||
      lastMessage.type === "frame_decode_error" ||
      lastMessage.type === "evaluate_process_error"
    ) {
      pendingFramesRef.current = Math.max(0, pendingFramesRef.current - 1);
      setPendingFrames(pendingFramesRef.current);
    }

    if (lastMessage.final_result) {
      setFinalResult(lastMessage.final_result);
      setCaptureEnabled(false);
    } else if (lastMessage.last_result) {
      setFinalResult(lastMessage.last_result);
    }
  }, [lastMessage]);

  useEffect(() => {
    if (!socketEnabled) {
      pendingFramesRef.current = 0;
      setPendingFrames(0);
      setCaptureEnabled(false);
    }
  }, [socketEnabled]);

  const handleFrame = useCallback((framePayload) => {
    if (pendingFramesRef.current >= MAX_PENDING_FRAMES) {
      return;
    }

    const sent = sendCameraFrame(framePayload);

    if (sent) {
      pendingFramesRef.current += 1;
      setPendingFrames(pendingFramesRef.current);
      setSentFrames((value) => value + 1);
    }
  }, [sendCameraFrame]);

  function handleClearResult() {
    setFinalResult(null);
    setSentFrames(0);
  }

  function handleResetEvaluate() {
    resetEvaluate();
    setFinalResult(null);
    setSentFrames(0);
    pendingFramesRef.current = 0;
    setPendingFrames(0);
  }

  return (
    <section className="camera-frame-diagnostics card">
      <div className="backend-diagnostics-header">
        <div>
          <h3>Diagnóstico Evaluate real</h3>
          <p>
            Prueba la máquina de estados de evaluación guiada con modelo,
            calidad, dinámica y geometría.
          </p>
        </div>

        <span className={connected ? "diag-status ok" : "diag-status error"}>
          {connected ? "Evaluate WS OK" : "Evaluate WS desconectado"}
        </span>
      </div>

      <div className="diag-actions">
        <label className="diag-inline-field">
          Letra objetivo
          <input
            value={targetLabel}
            maxLength={1}
            onChange={(event) => setTargetLabel(event.target.value.toUpperCase())}
            disabled={socketEnabled}
          />
        </label>

        <label className="diag-inline-field">
          Grabación
          <input
            type="number"
            min="1"
            step="0.5"
            value={autoStopSeconds}
            onChange={(event) => setAutoStopSeconds(Number(event.target.value))}
            disabled={socketEnabled}
          />
        </label>

        <button
          className="secondary"
          onClick={() => setSocketEnabled((value) => !value)}
        >
          {socketEnabled ? "Cerrar Evaluate WS" : "Abrir Evaluate WS"}
        </button>

        <button
          className="primary"
          onClick={() => setCaptureEnabled((value) => !value)}
          disabled={!connected}
        >
          {captureEnabled ? "Detener frames" : "Enviar frames"}
        </button>

        <button
          className="secondary"
          onClick={handleClearResult}
        >
          Limpiar resultado
        </button>

        <button
          className="secondary"
          onClick={handleResetEvaluate}
          disabled={!connected}
        >
          Reiniciar evaluate
        </button>
      </div>

      {error && <p className="diag-error">{error}</p>}

      <div className="camera-frame-grid">
        <CameraView
          title="Evaluación guiada"
          subtitle="Mostrá la letra objetivo cuando el estado lo indique."
          captureEnabled={connected && captureEnabled}
          captureFps={20}
          jpegQuality={0.65}
          maxCaptureWidth={640}
          mirrorPreview={false}
          mirrorCapture={false}
          onFrame={handleFrame}
        />

        <div className="diag-box">
          <strong>Estado de evaluación</strong>

          <p>
            Frames enviados: <strong>{sentFrames}</strong>
          </p>

          <p>
            Frames pendientes backend: <strong>{pendingFrames}</strong>
          </p>

          {lastMessage?.state && (
            <>
              <p>
                Estado: <strong>{lastMessage.state}</strong>
              </p>

              <p>
                Evento: <strong>{lastMessage.event || "-"}</strong>
              </p>

              <p>
                Manos detectadas:{" "}
                <strong>{lastMessage.live_detected_hands}</strong>
              </p>

              <p>
                Start counter:{" "}
                <strong>
                  {lastMessage.start_counter}/{lastMessage.start_consecutive_frames}
                </strong>
              </p>

              <p>
                Capturados:{" "}
                <strong>{lastMessage.captured_items_count}</strong>
              </p>
            </>
          )}

          {finalResult && (
            <div className="eval-final-box">
              <h4>Resultado final</h4>

              <p>
                Objetivo: <strong>{finalResult.target_label}</strong>
              </p>

              <p>
                Predicción: <strong>{finalResult.pred_label || "-"}</strong>
              </p>

              <p>
                Confianza:{" "}
                <strong>
                  {Number(finalResult.confidence || 0).toFixed(3)}
                </strong>
              </p>

              <p>
                Frames capturados:{" "}
                <strong>{finalResult.frames_captured}</strong>
              </p>

              <p>
                Frames muestreados:{" "}
                <strong>{finalResult.sampled_frames}</strong>
              </p>

              <p>
                Secuencia:{" "}
                <strong>
                  {Array.isArray(finalResult.sequence_shape)
                    ? finalResult.sequence_shape.join(" x ")
                    : "-"}
                </strong>
              </p>

              <p>
                Correcto:{" "}
                <strong>{finalResult.correct ? "sí" : "no"}</strong>
              </p>

              <p>
                Aceptado:{" "}
                <strong>{finalResult.accepted ? "sí" : "no"}</strong>
              </p>

              <p>{finalResult.quality_message}</p>
            </div>
          )}

          {lastMessage ? (
            <pre>{JSON.stringify(lastMessage, null, 2)}</pre>
          ) : (
            <p className="diag-error">
              Todavía no hay mensajes de evaluación.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}