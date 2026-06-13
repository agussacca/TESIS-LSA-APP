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
    if (!enabled) return;

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
        setLastMessage(data);
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
    });
  }, [sendJson]);

  const resetEvaluate = useCallback(() => {
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