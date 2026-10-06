const DEFAULT_WS_BASE_URL = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`;
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || DEFAULT_WS_BASE_URL;

export function createEvaluateTestSocket({
  targetLabel = "A",
  autoStopSeconds = 2.5,
  stabilizeSeconds = 1.0,
  startConsecutiveFrames = 10,
  releaseConsecutiveFrames = 8,
  onOpen,
  onMessage,
  onError,
  onClose,
} = {}) {
  const params = new URLSearchParams({
    target_label: targetLabel,
    auto_stop_seconds: String(autoStopSeconds),
    stabilize_seconds: String(stabilizeSeconds),
    start_consecutive_frames: String(startConsecutiveFrames),
    release_consecutive_frames: String(releaseConsecutiveFrames),
  });

  const socket = new WebSocket(`${WS_BASE_URL}/ws/evaluate/test?${params.toString()}`);

  socket.onopen = () => {
    onOpen?.();

    socket.send(JSON.stringify({
      type: "hello",
      source: "react-vite-evaluate",
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
