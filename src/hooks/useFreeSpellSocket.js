import { useCallback, useEffect, useRef, useState } from "react";
import { createFreeSpellSocket } from "../services/spellSocket";

export function useFreeSpellSocket({
  enabled = false,
  recordSeconds = 2.0,
  stabilizeSeconds = 1.0,
  startConsecutiveFrames = 10,
  releaseConsecutiveFrames = 8,
  startMinHands = 1,
  minConfidence = 0.70,
  minMargin = 0.15,
} = {}) {
  const socketRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!enabled) return;

    const socket = createFreeSpellSocket({
      recordSeconds,
      stabilizeSeconds,
      startConsecutiveFrames,
      releaseConsecutiveFrames,
      startMinHands,
      minConfidence,
      minMargin,
      onOpen: () => {
        setConnected(true);
        setError(null);
      },
      onMessage: (data) => {
        setLastMessage(data);
      },
      onError: () => {
        setError("Error en WebSocket de deletreo libre");
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
    recordSeconds,
    stabilizeSeconds,
    startConsecutiveFrames,
    releaseConsecutiveFrames,
    startMinHands,
    minConfidence,
    minMargin,
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

  const deleteLast = useCallback(() => {
    return sendJson({
      type: "delete_last",
      timestamp: Date.now(),
    });
  }, [sendJson]);

  const appendSpace = useCallback(() => {
    return sendJson({
      type: "append_space",
      timestamp: Date.now(),
    });
  }, [sendJson]);

  const clearText = useCallback(() => {
    return sendJson({
      type: "clear_text",
      timestamp: Date.now(),
    });
  }, [sendJson]);

  const resetState = useCallback(() => {
    return sendJson({
      type: "reset_state",
      timestamp: Date.now(),
    });
  }, [sendJson]);

  return {
    connected,
    lastMessage,
    error,
    sendCameraFrame,
    deleteLast,
    appendSpace,
    clearText,
    resetState,
  };
}