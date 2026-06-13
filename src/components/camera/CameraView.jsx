//CameraView.jsx
import { useEffect, useRef } from "react";
import { useCameraStream } from "../../hooks/useCameraStream";

export default function CameraView({
  title = "Vista de cámara",
  subtitle = "Activá la cámara para comenzar la práctica.",
  autoStart = false,
  captureEnabled = false,
  captureFps = 2,
  jpegQuality = 0.65,
  maxCaptureWidth = 640,
  mirrorPreview = false,
  mirrorCapture = false,
  onFrame,
}) {
  const canvasRef = useRef(null);

  const {
    videoRef,
    status,
    error,
    startCamera,
    stopCamera,
    isActive,
  } = useCameraStream({ autoStart });

  useEffect(() => {
    if (!captureEnabled || !isActive || !onFrame) return;

    const safeFps = Math.min(Math.max(captureFps, 1), 30);
    const intervalMs = 1000 / safeFps;

    const intervalId = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas) return;
      if (!video.videoWidth || !video.videoHeight) return;

      const originalWidth = video.videoWidth;
      const originalHeight = video.videoHeight;

      const scale = Math.min(1, maxCaptureWidth / originalWidth);
      const width = Math.round(originalWidth * scale);
      const height = Math.round(originalHeight * scale);

      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d");
      if (!context) return;

      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, width, height);

      if (mirrorCapture) {
        context.translate(width, 0);
        context.scale(-1, 1);
      }

      context.drawImage(video, 0, 0, width, height);

      context.setTransform(1, 0, 0, 1, 0, 0);

      const dataUrl = canvas.toDataURL("image/jpeg", jpegQuality);
      const imageBase64 = dataUrl.split(",")[1];

      onFrame({
        type: "camera_frame",
        image_format: "jpeg",
        image_base64: imageBase64,
        width,
        height,
        original_width: originalWidth,
        original_height: originalHeight,
        captured_at: Date.now(),
        mirrored: mirrorCapture,
        orientation: mirrorCapture ? "mirrored" : "original",
      });
    }, intervalMs);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [
    captureEnabled,
    captureFps,
    isActive,
    jpegQuality,
    maxCaptureWidth,
    onFrame,
    videoRef,
  ]);

  return (
    <div className="camera-view">
      <div className="camera-view-head">
        <div>
          <strong>{title}</strong>
          <small>{subtitle}</small>
        </div>

        <span className={`camera-status ${isActive ? "active" : "inactive"}`}>
          {isActive ? "Cámara activa" : "Cámara inactiva"}
        </span>
      </div>

      <div className="camera-video-frame">
        <video
          ref={videoRef}
          className="camera-video"
          style={{ transform: mirrorPreview ? "scaleX(-1)" : "none" }}
          autoPlay
          muted
          playsInline
        />

        {!isActive && (
          <div className="camera-placeholder">
            <span>📷</span>
            <p>
              {status === "requesting"
                ? "Solicitando acceso a cámara..."
                : "La cámara todavía no está activa."}
            </p>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="camera-capture-canvas" />

      {error && (
        <div className="camera-error">
          {error}
        </div>
      )}

      <div className="camera-controls">
        <button
          className="primary"
          onClick={startCamera}
          disabled={status === "requesting" || isActive}
        >
          Activar cámara
        </button>

        <button
          className="secondary"
          onClick={stopCamera}
          disabled={!isActive}
        >
          Detener cámara
        </button>
      </div>
    </div>
  );
}