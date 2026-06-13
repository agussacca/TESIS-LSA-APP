// UserPanelSummary.jsx
import { useEffect, useState } from "react";
import { obtenerPanelUsuario } from "../../services/progressApi";

export default function UserPanelSummary({ usuarioId, isGuest = false }) {
  const [panel, setPanel] = useState(null);
  const [status, setStatus] = useState("loading");

  async function loadPanel() {
    setStatus("loading");

    try {
      if (isGuest || !usuarioId) {
        setPanel({
          senias_aprendidas_camara: 0,
          palabras_deletreadas_exitosamente: 0,
          rondas_por_categoria: {},
          progreso_por_letra: [],
        });
        setStatus("ready");
        return;
      }

      const data = await obtenerPanelUsuario(usuarioId);
      setPanel(data);
      setStatus("ready");
    } catch (error) {
      console.error(error);
      setStatus("error");
    }
  }

  useEffect(() => {
    loadPanel();
  }, [usuarioId, isGuest]);

  if (status === "loading") {
    return (
      <section className="user-panel-summary card">
        <p className="progress-muted">Cargando resumen...</p>
      </section>
    );
  }

  if (status === "error") {
    return (
      <section className="user-panel-summary card">
        <p className="progress-error">No se pudo cargar el resumen.</p>
        <button className="secondary" onClick={loadPanel}>
          Reintentar
        </button>
      </section>
    );
  }

  const letrasConActividad = Array.isArray(panel?.progreso_por_letra)
    ? panel.progreso_por_letra.length
    : 0;
  const rondasExitosas = Object.values(panel?.rondas_por_categoria || {}).reduce(
    (total, value) => total + Number(value || 0),
    0
  );

  return (
    <section className="user-panel-summary card">
      <div className="user-panel-summary-head">
        <div>
          <small>Resumen real</small>
          <h3>Actividad del usuario</h3>
        </div>

        <button className="secondary" onClick={loadPanel}>
          Actualizar
        </button>
      </div>

      <div className="user-panel-summary-grid">
        <SummaryItem label="Señas aprendidas" value={panel?.senias_aprendidas_camara ?? 0} />
        <SummaryItem label="Palabras deletreadas" value={panel?.palabras_deletreadas_exitosamente ?? 0} />
        <SummaryItem label="Rondas exitosas" value={rondasExitosas} />
        <SummaryItem label="Letras con actividad" value={letrasConActividad} />
      </div>
    </section>
  );
}

function SummaryItem({ label, value }) {
  return (
    <div className="user-summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
