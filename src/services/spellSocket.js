const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || "ws://127.0.0.1:8000";

export function createFreeSpellSocket({
  recordSeconds = 2.0,
  stabilizeSeconds = 1.0,
  startConsecutiveFrames = 10,
  releaseConsecutiveFrames = 8,
  startMinHands = 1,
  minConfidence = 0.70,
  minMargin = 0.15,
  onOpen,
  onMessage,
  onError,
  onClose,
} = {}) {
  const params = new URLSearchParams({
    record_seconds: String(recordSeconds),
    stabilize_seconds: String(stabilizeSeconds),
    start_consecutive_frames: String(startConsecutiveFrames),
    release_consecutive_frames: String(releaseConsecutiveFrames),
    start_min_hands: String(startMinHands),
    min_confidence: String(minConfidence),
    min_margin: String(minMargin),
  });

  const socket = new WebSocket(`${WS_BASE_URL}/ws/spell/free?${params.toString()}`);

  socket.onopen = () => {
    onOpen?.();

    socket.send(JSON.stringify({
      type: "hello",
      source: "react-vite-free-spell",
      timestamp: Date.now(),
    }));
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage?.(data);
    } catch {
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