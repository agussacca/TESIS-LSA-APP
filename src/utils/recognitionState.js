export function getRecognitionStateMeta(state, { completed = false } = {}) {
  if (completed) {
    return {
      label: "Completado",
      className: "completed",
      icon: "✓",
    };
  }

  switch (state) {
    case "IDLE":
      return {
        label: "Esperando seña",
        className: "idle",
        icon: "●",
      };

    case "STABILIZING":
      return {
        label: "Estabilizando",
        className: "stabilizing",
        icon: "◔",
      };

    case "RECORDING":
      return {
        label: "Grabando intento",
        className: "recording",
        icon: "●",
      };

    case "WAIT_RELEASE":
      return {
        label: "Retirá las manos",
        className: "wait-release",
        icon: "↯",
      };

    default:
      return {
        label: "Sin iniciar",
        className: "inactive",
        icon: "○",
      };
  }
}