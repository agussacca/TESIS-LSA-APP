import { useEffect, useState } from "react";
import { getHealth } from "../../services/apiClient";
import { useRecognitionSocket } from "../../hooks/useRecognitionSocket";

export default function BackendDiagnostics() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [socketEnabled, setSocketEnabled] = useState(false);

  const {
    connected,
    lastMessage,
    error: socketError,
    sendTestFrame,
  } = useRecognitionSocket(socketEnabled);

  useEffect(() => {
    getHealth()
      .then((data) => {
        setHealth(data);
        setHealthError(null);
      })
      .catch((error) => {
        setHealth(null);
        setHealthError(error.message);
      });
  }, []);

  return (
    <section className="backend-diagnostics card">
      <div className="backend-diagnostics-header">
        <div>
          <h3>Diagnóstico de backend</h3>
          <p>Prueba temporal de conexión entre React/Vite y FastAPI.</p>
        </div>

        <span className={health?.status === "ok" ? "diag-status ok" : "diag-status error"}>
          {health?.status === "ok" ? "API OK" : "API sin conexión"}
        </span>
      </div>

      <div className="diag-grid">
        <div className="diag-box">
          <strong>REST /api/health</strong>

          {health ? (
            <pre>{JSON.stringify(health, null, 2)}</pre>
          ) : (
            <p className="diag-error">
              {healthError || "Consultando backend..."}
            </p>
          )}
        </div>

        <div className="diag-box">
          <strong>WebSocket /ws/recognition/test</strong>

          <div className="diag-actions">
            <button
              className="secondary"
              onClick={() => setSocketEnabled((value) => !value)}
            >
              {socketEnabled ? "Cerrar WebSocket" : "Abrir WebSocket"}
            </button>

            <button
              className="primary"
              onClick={sendTestFrame}
              disabled={!connected}
            >
              Enviar mensaje dummy
            </button>
          </div>

          <p>
            Estado:{" "}
            <span className={connected ? "diag-ok-text" : "diag-error-text"}>
              {connected ? "conectado" : "desconectado"}
            </span>
          </p>

          {socketError && (
            <p className="diag-error">{socketError}</p>
          )}

          {lastMessage && (
            <pre>{JSON.stringify(lastMessage, null, 2)}</pre>
          )}
        </div>
      </div>
    </section>
  );
}