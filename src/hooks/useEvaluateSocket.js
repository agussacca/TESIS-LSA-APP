//useEvaluateSocket.js
import { useCallback, useEffect, useRef, useState } from "react";
import { createEvaluateTestSocket } from "../services/evaluateSocket";

export function useEvaluateSocket({
  enabled = false,
  targetLabel = "A",
  autoStopSeconds = 2.0,
  stabilizeSeconds = 1.0,
  startConsecutiveFrames = 10,
  releaseConsecutiveFrames = 8,
} = {}) {
  const socketRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Cada evaluación debe comenzar sin conservar mensajes de la letra anterior.
    setLastMessage(null);
    setError(null);

    if (!enabled) {
      return;
    }

    const socket = createEvaluateTestSocket({
      targetLabel,
      autoStopSeconds,
      stabilizeSeconds,
      startConsecutiveFrames,
      releaseConsecutiveFrames,
      onOpen: () => {
        setConnected(true);
        setError(null);
      },
      onMessage: (data) => {
        // Marca el instante en que el navegador recibe efectivamente
        // el mensaje WebSocket, antes de que React procese/renderice el estado.
        setLastMessage({
          ...data,
          client_received_perf_ms: performance.now(),
        });
      },
      onError: () => {
        setError("Error en WebSocket de evaluación");
      },
      onClose: () => {
        setConnected(false);
      },
    });

    socketRef.current = socket;

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, [
    enabled,
    targetLabel,
    autoStopSeconds,
    stabilizeSeconds,
    startConsecutiveFrames,
    releaseConsecutiveFrames,
  ]);

  const sendJson = useCallback((payload) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    socketRef.current.send(JSON.stringify(payload));
    return true;
  }, []);

  const sendCameraFrame = useCallback((framePayload) => {
    return sendJson({
      ...framePayload,
      type: "camera_frame",
      timestamp: Date.now(),

      // Marca temporal de alta resolución generada por el navegador.
      // Se devuelve sin modificar desde el backend y permite medir
      // el tiempo transcurrido hasta recibir el resultado final.
      client_sent_perf_ms: performance.now(),
    });
  }, [sendJson]);

  const resetEvaluate = useCallback(() => {
    // El reset remoto también invalida localmente cualquier resultado previo.
    setLastMessage(null);
    setError(null);

    return sendJson({
      type: "reset_evaluate",
      timestamp: Date.now(),
    });
  }, [sendJson]);

  return {
    connected,
    lastMessage,
    error,
    sendCameraFrame,
    resetEvaluate,
  };
}
