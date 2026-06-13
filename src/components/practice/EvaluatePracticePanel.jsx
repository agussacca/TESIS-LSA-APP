//EvaluatePracticePanel.jsx
import { useCallback, useEffect, useRef, useState } from "react";
import CameraView from "../camera/CameraView";
import { useEvaluateSocket } from "../../hooks/useEvaluateSocket";
import { getRecognitionStateMeta } from "../../utils/recognitionState";
import { guardarIntentoPractica } from "../../services/progressApi";

const MAX_PENDING_FRAMES = 80;

export default function EvaluatePracticePanel({
  targetLabel,
  onNext,
  isLast = false,
  onGamificationSync,
  usuarioId = null,
  persistEnabled = true,
}) {
  const [socketEnabled, setSocketEnabled] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [finalResult, setFinalResult] = useState(null);
  const [sentFrames, setSentFrames] = useState(0);
  const [pendingFrames, setPendingFrames] = useState(0);
  const [lastImportantEvent, setLastImportantEvent] = useState(null);
  const [saveStatus, setSaveStatus] = useState("idle");

  const pendingFramesRef = useRef(0);
  const savedAttemptKeyRef = useRef(null);

  const {
    connected,
    lastMessage,
    error,
    sendCameraFrame,
    resetEvaluate,
  } = useEvaluateSocket({
    enabled: socketEnabled,
    targetLabel,
    autoStopSeconds: 2.0,
    stabilizeSeconds: 1.0,
    startConsecutiveFrames: 10,
    releaseConsecutiveFrames: 8,
  });

  const persistFinalResult = useCallback(async (result, messageCounter) => {
    if (!result) return;

    const attemptKey = [
      "practica_letra",
      messageCounter ?? "sin_counter",
      result.target_label,
      result.pred_label,
      result.confidence,
      result.accepted,
    ].join("|");

    if (savedAttemptKeyRef.current === attemptKey) {
      return;
    }

    savedAttemptKeyRef.current = attemptKey;
    setSaveStatus("saving");

    if (!persistEnabled || !usuarioId) {
      setSaveStatus("idle");
      return;
    }

    try {
      await guardarIntentoPractica({
        usuario_id: usuarioId,
        letra_esperada: result.target_label || targetLabel,
        letra_predicha: result.pred_label,
        validado: Boolean(result.accepted),
      });

      setSaveStatus("saved");

      try {
        await onGamificationSync?.();
      } catch (syncError) {
        console.warn(
          "El intento se guardó, pero no se pudo sincronizar gamificación:",
          syncError
        );
      }
    } catch (error) {
      console.error(error);
      setSaveStatus("error");
    }
  }, [onGamificationSync, persistEnabled, targetLabel, usuarioId]);

  useEffect(() => {
    setSocketEnabled(false);
    setCaptureEnabled(false);
    setFinalResult(null);
    setSentFrames(0);
    setPendingFrames(0);
    setLastImportantEvent(null);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedAttemptKeyRef.current = null;
  }, [targetLabel]);

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

    if (lastMessage.event) {
      setLastImportantEvent(lastMessage.event);
    }

    if (lastMessage.final_result) {
      setFinalResult(lastMessage.final_result);
      setCaptureEnabled(false);
      persistFinalResult(lastMessage.final_result, lastMessage.counter);
    } else if (lastMessage.last_result) {
      setFinalResult(lastMessage.last_result);
    }
  }, [lastMessage, persistFinalResult]);

  useEffect(() => {
    if (!socketEnabled) {
      setCaptureEnabled(false);
      pendingFramesRef.current = 0;
      setPendingFrames(0);
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

  function resetAttemptUiState() {
    setFinalResult(null);
    setLastImportantEvent(null);
    setSentFrames(0);
    setPendingFrames(0);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedAttemptKeyRef.current = null;
  }

  function startAttempt() {
    resetAttemptUiState();

    if (connected) {
      resetEvaluate();
    }

    setSocketEnabled(true);
    setCaptureEnabled(true);
  }

  function stopAttempt() {
    setCaptureEnabled(false);
  }

  function retryAttempt() {
    resetAttemptUiState();

    if (connected) {
      resetEvaluate();
    } else {
      setSocketEnabled(true);
    }

    setCaptureEnabled(true);
  }

  function goNext() {
    setCaptureEnabled(false);
    setSocketEnabled(false);
    onNext?.();
  }

  const accepted = finalResult?.accepted === true;
  const stateMeta = getRecognitionStateMeta(lastMessage?.state);

  return (
    <div className="camera-pro-layout">
      <section className="camera-main card fade-up">
        <div className="camera-main-head">
          <small>Vista de cámara</small>
          <div className="pill blue-pill">
            {connected ? "Evaluación conectada" : "Tiempo real"}
          </div>
        </div>

        <CameraView
          title="Vista de cámara"
          subtitle="Usá buena iluminación y mantené la mano dentro del encuadre."
          captureEnabled={connected && captureEnabled}
          captureFps={20}
          jpegQuality={0.65}
          maxCaptureWidth={640}
          mirrorPreview={false}
          mirrorCapture={false}
          onFrame={handleFrame}
        />
      </section>

      <section className="camera-side card fade-up delay-1">
        <small>Letra actual</small>
        <div className="big-letter">{targetLabel}</div>

        <div className="camera-side-note">
          <p>Activá la cámara, iniciá la evaluación y realizá la seña indicada.</p>
        </div>

        <div className="evaluate-status-box">
          <div>
            <span>Estado</span>
            <strong className={`state-badge ${stateMeta.className}`}>
              <span>{stateMeta.icon}</span>
              {stateMeta.label}
            </strong>
          </div>

          <div>
            <span>Manos detectadas</span>
            <strong>{lastMessage?.live_detected_hands ?? "-"}</strong>
          </div>

          <div>
            <span>Frames enviados</span>
            <strong>{sentFrames}</strong>
          </div>

          <div>
            <span>Pendientes</span>
            <strong>{pendingFrames}</strong>
          </div>

          {lastImportantEvent && (
            <div>
              <span>Último evento</span>
              <strong>{getFriendlyEvent(lastImportantEvent)}</strong>
            </div>
          )}
        </div>

        {error && (
          <div className="result-banner incorrecto">
            Error de conexión con evaluación.
          </div>
        )}

        {finalResult && (
          <div className={`result-banner ${accepted ? "correcto" : "incorrecto"}`}>
            {accepted ? "✔ Correcto" : "✖ Incorrecto"}
          </div>
        )}

        {finalResult && (
          <div className="evaluate-result-detail">
            <p>
              Letra esperada:{" "}
              <strong>{finalResult.target_label || targetLabel}</strong>
            </p>

            <p>
              Letra detectada: <strong>{finalResult.pred_label || "-"}</strong>
            </p>

            <p>
              Confianza:{" "}
              <strong>{Number(finalResult.confidence || 0).toFixed(3)}</strong>
            </p>

            <p className={accepted ? "eval-ok-text" : "eval-error-text"}>
              {accepted
                ? "El intento fue aceptado."
                : "El intento no fue aceptado. Probá nuevamente."}
            </p>

            {saveStatus === "saving" && (
              <small className="practice-save-status">
                Guardando intento...
              </small>
            )}

            {saveStatus === "saved" && (
              <small className="practice-save-status ok">
                Intento guardado.
              </small>
            )}

            {saveStatus === "error" && (
              <small className="practice-save-status error">
                No se pudo guardar el intento.
              </small>
            )}
          </div>
        )}

        <div className="action-column">
          {!captureEnabled && !finalResult && (
            <button className="primary" onClick={startAttempt}>
              Iniciar evaluación
            </button>
          )}

          {captureEnabled && (
            <button className="secondary" onClick={stopAttempt}>
              Detener envío
            </button>
          )}

          {finalResult && (
            <button className="secondary" onClick={retryAttempt}>
              Reintentar
            </button>
          )}

          <button
            className="primary"
            onClick={goNext}
            disabled={!finalResult || isLast}
          >
            Siguiente
          </button>
        </div>

        {isLast && finalResult && (
          <small className="practice-end-note">
            Llegaste al final de esta práctica.
          </small>
        )}
      </section>
    </div>
  );
}

function getFriendlyEvent(event) {
  switch (event) {
    case "stabilizing_started":
      return "Estabilización iniciada";
    case "stabilizing_cancelled":
      return "Estabilización cancelada";
    case "recording_started":
      return "Grabación iniciada";
    case "attempt_finalized":
      return "Intento finalizado";
    case "released_ready_for_next_attempt":
      return "Listo para otro intento";
    default:
      return event;
  }
}