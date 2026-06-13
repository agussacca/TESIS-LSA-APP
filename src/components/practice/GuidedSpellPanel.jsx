//GuidedSpellPanel.jsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CameraView from "../camera/CameraView";
import { useEvaluateSocket } from "../../hooks/useEvaluateSocket";
import { getRecognitionStateMeta } from "../../utils/recognitionState";
import { guardarIntentoPractica, registrarPalabraDeletreada } from "../../services/progressApi";

const MAX_PENDING_FRAMES = 80;

function normalizeWord(word) {
  return String(word || "")
    .trim()
    .toUpperCase()
    .split("")
    .filter((char) => char.trim().length > 0);
}

export default function GuidedSpellPanel({
  word = "SOL",
  onCompleted,
  onGamificationSync,
  usuarioId = null,
  persistEnabled = true,
}) {
  const letters = useMemo(() => normalizeWord(word), [word]);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [socketEnabled, setSocketEnabled] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [finalResult, setFinalResult] = useState(null);
  const [sentFrames, setSentFrames] = useState(0);
  const [pendingFrames, setPendingFrames] = useState(0);
  const [lastImportantEvent, setLastImportantEvent] = useState(null);
  const [completedResults, setCompletedResults] = useState([]);
  const [wordCompleted, setWordCompleted] = useState(false);
  const [saveStatus, setSaveStatus] = useState("idle");

  const pendingFramesRef = useRef(0);
  const savedGuidedAttemptKeyRef = useRef(null);

  const currentLetter = letters[currentIndex] || null;

  const completedCount = wordCompleted
    ? letters.length
    : Math.min(
        completedResults.length + (finalResult?.accepted ? 1 : 0),
        letters.length
      );

  const progressPercentage =
    letters.length > 0 ? (completedCount / letters.length) * 100 : 0;

  const {
    connected,
    lastMessage,
    error,
    sendCameraFrame,
    resetEvaluate,
  } = useEvaluateSocket({
    enabled: socketEnabled && Boolean(currentLetter),
    targetLabel: currentLetter || "A",
    autoStopSeconds: 2.0,
    stabilizeSeconds: 1.0,
    startConsecutiveFrames: 10,
    releaseConsecutiveFrames: 8,
  });

  const persistGuidedSpellResult = useCallback(async (result, messageCounter) => {
    if (!result || !currentLetter) return;

    const attemptKey = [
      "deletreo_guiado",
      messageCounter ?? "sin_counter",
      word,
      currentIndex,
      currentLetter,
      result.pred_label,
      result.confidence,
      result.accepted,
    ].join("|");

    if (savedGuidedAttemptKeyRef.current === attemptKey) {
      return;
    }

    savedGuidedAttemptKeyRef.current = attemptKey;
    setSaveStatus("saving");

    if (!persistEnabled || !usuarioId) {
      setSaveStatus("idle");
      return;
    }

    try {
      await guardarIntentoPractica({
        usuario_id: usuarioId,
        letra_esperada: currentLetter,
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
  }, [currentLetter, onGamificationSync, persistEnabled, usuarioId]);

  useEffect(() => {
    setCurrentIndex(0);
    setSocketEnabled(false);
    setCaptureEnabled(false);
    setFinalResult(null);
    setSentFrames(0);
    setPendingFrames(0);
    setLastImportantEvent(null);
    setCompletedResults([]);
    setWordCompleted(false);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedGuidedAttemptKeyRef.current = null;
  }, [word]);

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
      persistGuidedSpellResult(lastMessage.final_result, lastMessage.counter);
    } else if (lastMessage.last_result) {
      setFinalResult(lastMessage.last_result);
    }
  }, [lastMessage, persistGuidedSpellResult]);

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

  function resetLetterState() {
    setFinalResult(null);
    setSentFrames(0);
    setPendingFrames(0);
    setLastImportantEvent(null);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedGuidedAttemptKeyRef.current = null;
  }

  function startLetterAttempt() {
    resetLetterState();

    if (connected) {
      resetEvaluate();
    }

    setSocketEnabled(true);
    setCaptureEnabled(true);
  }

  function stopLetterAttempt() {
    setCaptureEnabled(false);
  }

  function retryLetter() {
    resetLetterState();

    if (connected) {
      resetEvaluate();
    } else {
      setSocketEnabled(true);
    }

    setCaptureEnabled(true);
  }

  async function advanceLetter() {
    if (!finalResult?.accepted) return;

    const nextResults = [...completedResults, finalResult];
    setCompletedResults(nextResults);

    setCaptureEnabled(false);
    setSocketEnabled(false);
    resetLetterState();

    if (currentIndex >= letters.length - 1) {
      setWordCompleted(true);

      if (persistEnabled && usuarioId) {
        try {
          setSaveStatus("saving");
          await registrarPalabraDeletreada({
            usuario_id: usuarioId,
            palabra: word,
          });
          setSaveStatus("saved");
          await onGamificationSync?.();
        } catch (error) {
          console.error("No se pudo registrar la palabra deletreada:", error);
          setSaveStatus("error");
        }
      }

      onCompleted?.({
        word,
        results: nextResults,
      });
      return;
    }

    setCurrentIndex((value) => value + 1);
  }

  function restartWord() {
    setCurrentIndex(0);
    setSocketEnabled(false);
    setCaptureEnabled(false);
    setFinalResult(null);
    setSentFrames(0);
    setPendingFrames(0);
    setLastImportantEvent(null);
    setCompletedResults([]);
    setWordCompleted(false);
    setSaveStatus("idle");

    pendingFramesRef.current = 0;
    savedGuidedAttemptKeyRef.current = null;
  }

  const accepted = finalResult?.accepted === true;
  const rejected = finalResult && !accepted;

  const stateMeta = getRecognitionStateMeta(lastMessage?.state, {
    completed: wordCompleted,
  });

  return (
    <section className="spell-layout">
      <div className="spell-main card fade-up">
        <div className="spell-top">
          <small>Palabra objetivo</small>
          <strong>
            {completedCount}/{letters.length} letras completadas
          </strong>
        </div>

        <h3>{word.toUpperCase()}</h3>

        <div className="guided-spell-letters">
          {letters.map((letter, index) => (
            <span
              key={`${letter}-${index}`}
              className={[
                "guided-spell-letter",
                wordCompleted || index < currentIndex ? "done" : "",
                index === currentIndex && !wordCompleted && !accepted ? "active" : "",
                index === currentIndex && accepted ? "accepted" : "",
              ].join(" ")}
            >
              {letter}
            </span>
          ))}
        </div>

        <div className="progress slim">
          <div style={{ width: `${progressPercentage}%` }} />
        </div>

        <CameraView
          title="Cámara de deletreo guiado"
          subtitle="Realizá la letra actual para avanzar en la palabra."
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
        <div className="pill gold-pill">
          {wordCompleted ? "Palabra completa" : "Letra actual"}
        </div>

        <div className="big-letter compact">
          {wordCompleted ? "✓" : currentLetter}
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
            Error de conexión con evaluación.
          </div>
        )}

        {finalResult && (
          <div className={`result-banner ${accepted ? "correcto" : "incorrecto"}`}>
            {accepted ? "✔ Letra correcta" : "✖ Reintentar letra"}
          </div>
        )}

        {finalResult && (
          <div className="guided-spell-result">
            <p>
              Esperada: <strong>{currentLetter}</strong>
            </p>

            <p>
              Detectada: <strong>{finalResult.pred_label || "-"}</strong>
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

        {wordCompleted && (
          <div className="guided-spell-completed">
            <strong>Palabra completada</strong>
            <p>Terminaste el deletreo guiado de {word.toUpperCase()}.</p>
          </div>
        )}

        <div className="action-column">
          {!captureEnabled && !finalResult && !wordCompleted && (
            <button className="primary" onClick={startLetterAttempt}>
              Iniciar letra
            </button>
          )}

          {captureEnabled && (
            <button className="secondary" onClick={stopLetterAttempt}>
              Detener envío
            </button>
          )}

          {rejected && (
            <button className="secondary" onClick={retryLetter}>
              Reintentar
            </button>
          )}

          {accepted && !wordCompleted && (
            <button className="primary" onClick={advanceLetter}>
              {currentIndex >= letters.length - 1
                ? "Finalizar palabra"
                : "Siguiente letra"}
            </button>
          )}

          {wordCompleted && (
            <button className="secondary" onClick={restartWord}>
              Practicar otra vez
            </button>
          )}
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
      return "Grabación iniciada";
    case "attempt_finalized":
      return "Intento finalizado";
    case "released_ready_for_next_attempt":
      return "Listo para otro intento";
    default:
      return event;
  }
}