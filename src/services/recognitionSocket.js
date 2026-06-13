const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || "ws://127.0.0.1:8000";

export function createRecognitionTestSocket({ onOpen, onMessage, onError, onClose } = {}) {
  const socket = new WebSocket(`${WS_BASE_URL}/ws/recognition/test`);

  socket.onopen = () => {
    onOpen?.();

    socket.send(JSON.stringify({
      type: "hello",
      source: "react-vite",
      timestamp: Date.now(),
    }));
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage?.(data);
    } catch (error) {
      console.error("Mensaje WebSocket inválido:", event.data);
    }
  };

  socket.onerror = (event) => {
    onError?.(event);
  };

  socket.onclose = () => {
    onClose?.();
  };

  return socket;
}