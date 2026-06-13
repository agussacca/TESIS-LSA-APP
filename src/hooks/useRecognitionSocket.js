import { useCallback, useEffect, useRef, useState } from "react";
import { createRecognitionTestSocket } from "../services/recognitionSocket";

export function useRecognitionSocket(enabled = false) {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!enabled) return;

    const socket = createRecognitionTestSocket({
      onOpen: () => {
        setConnected(true);
        setError(null);
      },
      onMessage: (data) => {
        setLastMessage(data);
      },
      onError: () => {
        setError("Error en WebSocket");
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
  }, [enabled]);

  const sendJson = useCallback((payload) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    socketRef.current.send(JSON.stringify(payload));
    return true;
  }, []);

  const sendTestFrame = useCallback(() => {
    return sendJson({
      type: "frame_dummy",
      timestamp: Date.now(),
    });
  }, [sendJson]);

  const sendCameraFrame = useCallback((framePayload) => {
    return sendJson({
      ...framePayload,
      type: "camera_frame",
      timestamp: Date.now(),
    });
  }, [sendJson]);

  return {
    connected,
    lastMessage,
    error,
    sendTestFrame,
    sendCameraFrame,
  };
}