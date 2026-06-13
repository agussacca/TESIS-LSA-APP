//FreeSpellPanel.jsx
import { useCallback, useEffect, useRef, useState } from "react";
import CameraView from "../camera/CameraView";
import { useFreeSpellSocket } from "../../hooks/useFreeSpellSocket";
import { getRecognitionStateMeta } from "../../utils/recognitionState";

const MAX_PENDING_FRAMES = 80;

export default function FreeSpellPanel({
  onGamificationSync,
  persistEnabled = false,
}) {
  const [socketEnabled, setSocketEnabled] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [sentFrames, setSentFrames] = useState(0);
  const [pendingFrames, setPendingFrames] = useState(0);
  const [spelledText, setSpelledText] = useState("");
  const [lastResult, setLastResult] = useState(null);
  const [lastImportantEvent, setLastImportantEvent] = useState(null);
  const [saveStatus, setSaveStatus] = useState("idle");

  const pendingFramesRef = useRef(0);
  const savedFreeAttemptKeyRef = useRef(null);

  const {
    connected,
    lastMessage,
    error,
    sendCameraFrame,
    deleteLast,
    appendSpace,
    clearText,
    resetState,
  } = useFreeSpellSocket({
    enabled: socketEnabled,
    recordSeconds: 2.0,
    stabilizeSeconds: 1.0,
    startConsecutiveFrames: 10,
    releaseConsecutiveFrames: 8,
    startMinHands: 1,
    minConfidence: 0.70,
    minMargin: 0.15,
  });

  const persistFreeSpellResult = useCallback(async (result, messageCounter) => {
    if (!result) return;

    // El deletreo libre no tiene una letra esperada objetiva.
    // Por decisión de diseño, no se persiste como intento de práctica ni como palabra exitosa.
    setSaveStatus("idle");
  }, []);

  useEffect(() => {
    if (!lastMessage) return;

    if (
      lastMessage.type === "free_spell_update" ||
      lastMessage.type === "frame_decode_error" ||
      lastMessage.type === "free_spell_process_error"
    ) {
      pendingFramesRef.current = Math.max(0, pendingFramesRef.current - 1);
      setPendingFrames(pendingFramesRef.current);
    }

    if (typeof lastMessage.spelled_text === "string") {
      setSpelledText(lastMessage.spelled_text);
    }

    if (lastMessage.event) {
      setLastImportantEvent(lastMessage.event);
    }

    if (lastMessage.final_result) {
      setLastResult(lastMessage.final_result);
      persistFreeSpellResult(lastMessage.final_result, lastMessage.counter);
    } else if (lastMessage.last_result) {
      setLastResult(lastMessage.last_result);
    }
  }, [lastMessage, persistFreeSpellResult]);

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

  function startFreeSpell() {
    setSocketEnabled(true);
    setCaptureEnabled(true);
    setSentFrames(0);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedFreeAttemptKeyRef.current = null;
    setPendingFrames(0);
  }

  function stopFreeSpell() {
    setCaptureEnabled(false);
  }

  function closeFreeSpell() {
    setCaptureEnabled(false);
    setSocketEnabled(false);
    setSentFrames(0);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedFreeAttemptKeyRef.current = null;
    setPendingFrames(0);
  }

  function handleDeleteLast() {
    deleteLast();
  }

  function handleAppendSpace() {
    appendSpace();
  }

  function handleClearText() {
    clearText();
    setSpelledText("");
    setLastResult(null);
    setSaveStatus("idle");

    savedFreeAttemptKeyRef.current = null;
  }

  function handleResetState() {
    resetState();
    setLastImportantEvent(null);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedFreeAttemptKeyRef.current = null;
    setPendingFrames(0);
  }

  const accepted = lastResult?.accepted === true;
  const stateMeta = getRecognitionStateMeta(lastMessage?.state);

  return (
    <section className="spell-layout">
      <div className="spell-main card fade-up">
        <div className="spell-top">
          <small>Texto detectado</small>
          <strong>
            {connected ? "Deletreo libre activo" : "Sin conexión"}
          </strong>
        </div>

        <div className="free-spell-output">
          {spelledText || <span>El texto aparecerá acá...</span>}
        </div>

        <CameraView
          title="Cámara de deletreo libre"
          subtitle="Realizá una letra, retirás la mano y luego podés realizar otra."
          captureEnabled={connected && captureEnabled}
          captureFps={20}
          jpegQuality={0.65}
          maxCaptureWidth={640}
          mirrorPreview={false}
          mirrorCapture={false}
          onFrame={handleFrame}
        />
      </div>

      <div className="spell-side card fade-up delay-1">
        <div className="pill gold-pill">Modo libre</div>

        <div className="big-letter compact">
          {lastResult?.pred_label || "?"}
        </div>

        <div className="guided-spell-status">
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
            Error de conexión con deletreo libre.
          </div>
        )}

        {lastResult && (
          <div className={`result-banner ${accepted ? "correcto" : "incorrecto"}`}>
            {accepted ? "✔ Letra agregada" : "✖ Letra no agregada"}
          </div>
        )}

        {lastResult && (
          <div className="guided-spell-result">
            <p>
              Predicción: <strong>{lastResult.pred_label || "-"}</strong>
            </p>

            <p>
              Confianza:{" "}
              <strong>{Number(lastResult.confidence || 0).toFixed(3)}</strong>
            </p>

            <p className={accepted ? "eval-ok-text" : "eval-error-text"}>
              {accepted
                ? "La letra fue agregada al texto."
                : "La letra no fue agregada. Probá nuevamente."}
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
          {!socketEnabled && (
            <button className="primary" onClick={startFreeSpell}>
              Iniciar deletreo libre
            </button>
          )}

          {socketEnabled && !captureEnabled && (
            <button className="primary" onClick={() => setCaptureEnabled(true)}>
              Reanudar envío
            </button>
          )}

          {captureEnabled && (
            <button className="secondary" onClick={stopFreeSpell}>
              Pausar envío
            </button>
          )}

          {socketEnabled && (
            <button className="secondary" onClick={closeFreeSpell}>
              Cerrar modo libre
            </button>
          )}

          <button
            className="secondary"
            onClick={handleDeleteLast}
            disabled={!connected}
          >
            Borrar última
          </button>

          <button
            className="secondary"
            onClick={handleAppendSpace}
            disabled={!connected}
          >
            Agregar espacio
          </button>

          <button
            className="secondary"
            onClick={handleClearText}
            disabled={!connected}
          >
            Limpiar texto
          </button>

          <button
            className="secondary"
            onClick={handleResetState}
            disabled={!connected}
          >
            Reiniciar estado
          </button>
        </div>
      </div>
    </section>
  );
}

function getFriendlyEvent(event) {
  switch (event) {
    case "stabilizing_started":
      return "Estabilización iniciada";
    case "stabilizing_cancelled":
      return "Estabilización cancelada";
    case "recording_started":
      return "Captura iniciada";
    case "attempt_finalized":
      return "Letra evaluada";
    case "released_ready_for_next_letter":
      return "Listo para otra letra";
    default:
      return event;
  }
}