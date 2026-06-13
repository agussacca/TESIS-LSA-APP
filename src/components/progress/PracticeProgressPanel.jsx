// PracticeProgressPanel.jsx
import { useEffect, useState } from "react";
import { obtenerPanelUsuario } from "../../services/progressApi";

const CATEGORIAS_ESTADISTICAS = [
  "Abecedario",
  "Comunicación Básica",
  "Familia",
  "Colores",
  "Números",
  "Deportes",
  "Provincias",
];

const EMPTY_SUMMARY = {
  senias_aprendidas_camara: 0,
  palabras_deletreadas_exitosamente: 0,
  rondas_por_categoria: {},
  progreso_por_letra: [],
};

export default function PracticeProgressPanel({ usuarioId, isGuest = false }) {
  const [summary, setSummary] = useState(EMPTY_SUMMARY);
  const [letters, setLetters] = useState([]);
  const [status, setStatus] = useState("loading");

  async function loadProgress() {
    setStatus("loading");

    try {
      if (isGuest || !usuarioId) {
        setSummary(EMPTY_SUMMARY);
        setLetters([]);
        setStatus("ready");
        return;
      }

      const data = await obtenerPanelUsuario(usuarioId);
      const nextSummary = data || EMPTY_SUMMARY;

      setSummary(nextSummary);
      setLetters(
        Array.isArray(nextSummary.progreso_por_letra)
          ? nextSummary.progreso_por_letra
          : []
      );
      setStatus("ready");
    } catch (error) {
      console.error(error);
      setStatus("error");
    }
  }

  useEffect(() => {
    loadProgress();
  }, [usuarioId, isGuest]);

  if (status === "loading") {
    return (
      <section className="progress-panel card fade-up">
        <h3>Tu progreso</h3>
        <p className="progress-muted">Cargando estadísticas...</p>
      </section>
    );
  }

  if (status === "error") {
    return (
      <section className="progress-panel card fade-up">
        <h3>Tu progreso</h3>
        <p className="progress-error">No se pudo cargar el progreso.</p>
        <button className="secondary" onClick={loadProgress}>
          Reintentar
        </button>
      </section>
    );
  }

  const rondasPorCategoria = summary?.rondas_por_categoria || {};

  return (
    <section className="progress-panel card fade-up">
      <div className="progress-panel-head">
        <div>
          <small>Estadísticas personales</small>
          <h3>Tu progreso</h3>
        </div>

        <button className="secondary" onClick={loadProgress}>
          Actualizar
        </button>
      </div>

      <div className="progress-summary-grid">
        <ProgressStat
          label="Señas aprendidas con cámara"
          value={summary?.senias_aprendidas_camara ?? 0}
        />

        {CATEGORIAS_ESTADISTICAS.map((categoria) => (
          <ProgressStat
            key={categoria}
            label={`Rondas de ${categoria} completadas con éxito`}
            value={rondasPorCategoria[categoria] ?? 0}
          />
        ))}

        <ProgressStat
          label="Cantidad de Palabras Deletreadas con éxito"
          value={summary?.palabras_deletreadas_exitosamente ?? 0}
        />
      </div>

      <div className="progress-by-letter">
        <div className="progress-section-title">
          <strong>Precisión por letra</strong>
          <small>{letters.length} letras con actividad</small>
        </div>

        {letters.length === 0 ? (
          <p className="progress-muted">
            Todavía no hay intentos registrados por letra.
          </p>
        ) : (
          <div className="letter-progress-list">
            {letters.map((item) => (
              <LetterProgressItem key={item.letra} item={item} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function ProgressStat({ label, value }) {
  return (
    <div className="progress-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LetterProgressItem({ item }) {
  const precision = item.precision ?? 0;
  const percent = Math.round(precision * 100);

  return (
    <div className="letter-progress-item">
      <div className="letter-progress-main">
        <strong>{item.letra}</strong>

        <div>
          <span>{percent}%</span>
          <small>
            {item.intentos_aceptados}/{item.total_intentos} aceptados
          </small>
        </div>
      </div>

      <div className="letter-progress-bar">
        <div style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
