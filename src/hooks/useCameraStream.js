import { useCallback, useEffect, useRef, useState } from "react";

export function useCameraStream({ autoStart = false } = {}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setStatus("stopped");
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      setStatus("requesting");

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("El navegador no soporta acceso a cámara mediante getUserMedia.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setStatus("active");
    } catch (err) {
      console.error("No se pudo iniciar la cámara:", err);
      setError(err?.message || "No se pudo iniciar la cámara.");
      setStatus("error");
      stopCamera();
    }
  }, [stopCamera]);

  useEffect(() => {
    if (autoStart) {
      startCamera();
    }

    return () => {
      stopCamera();
    };
  }, [autoStart, startCamera, stopCamera]);

  return {
    videoRef,
    stream: streamRef.current,
    status,
    error,
    startCamera,
    stopCamera,
    isActive: status === "active",
  };
}