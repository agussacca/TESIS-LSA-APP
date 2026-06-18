// progressApi.js
import { getAuthToken } from "../utils/authSession";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function buildUrl(path) {
  return `${API_BASE_URL}${path}`;
}

async function requestJson(path, options = {}) {
  const headers = new Headers(options.headers || {});

  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const token = getAuthToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(buildUrl(path), {
    ...options,
    headers,
  });

  if (!response.ok) {
    const text = await response.text();
    const error = new Error(`${response.status} ${text}`);
    error.status = response.status;
    error.body = text;
    throw error;
  }

  return response.json();
}

function normalizeUserId(usuarioId) {
  const value = Number(usuarioId);
  return Number.isFinite(value) && value > 0 ? value : null;
}

export async function crearUsuarioRegistrado(payload) {
  return requestJson("/api/usuarios", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function obtenerUsuarioRegistrado(usuarioId) {
  const id = normalizeUserId(usuarioId);

  if (!id) {
    return null;
  }

  return requestJson(`/api/usuarios/${encodeURIComponent(id)}`);
}

export async function guardarIntentoPractica(payload) {
  const usuarioId = normalizeUserId(payload?.usuario_id);

  if (!usuarioId) {
    return { omitido: true, motivo: "usuario_no_registrado" };
  }

  return requestJson("/api/intentos-practica", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      usuario_id: usuarioId,
      letra_esperada: payload.letra_esperada,
      letra_predicha: payload.letra_predicha,
      validado: Boolean(payload.validado),
    }),
  });
}

export async function registrarPalabraDeletreada(payload) {
  const usuarioId = normalizeUserId(payload?.usuario_id);

  if (!usuarioId) {
    return { omitido: true, motivo: "usuario_no_registrado" };
  }

  return requestJson("/api/palabras-deletreadas", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      usuario_id: usuarioId,
      palabra: payload.palabra,
    }),
  });
}

export async function obtenerResumenPractica(usuarioId) {
  return obtenerPanelUsuario(usuarioId);
}

export async function obtenerProgresoLetras(usuarioId) {
  const id = normalizeUserId(usuarioId);
  if (!id) return [];
  return requestJson(`/api/progreso-letras/${encodeURIComponent(id)}`);
}

export async function obtenerPanelUsuario(usuarioId) {
  const id = normalizeUserId(usuarioId);

  if (!id) {
    return {
      senias_aprendidas_camara: 0,
      palabras_deletreadas_exitosamente: 0,
      rondas_por_categoria: {},
      progreso_por_letra: [],
      progreso: {
        xp_total: 0,
        nivel: 1,
        racha_actual: 0,
        racha_maxima: 0,
      },
    };
  }

  return requestJson(`/api/panel-usuario/${encodeURIComponent(id)}`);
}

export async function obtenerObjetivosUsuario(usuarioId) {
  const id = normalizeUserId(usuarioId);

  if (!id) {
    return {
      diarios: [],
      semanales: [],
      progreso: null,
    };
  }

  try {
    return await requestJson(`/api/objetivos-usuario/${encodeURIComponent(id)}`);
  } catch (error) {
    if (error.status !== 404) {
      throw error;
    }

    const panel = await obtenerPanelUsuario(id);
    return {
      diarios: [],
      semanales: [],
      progreso: panel.progreso,
    };
  }
}

export async function sincronizarGamificacionUsuario(usuarioId) {
  const id = normalizeUserId(usuarioId);

  if (!id) {
    return {
      progreso: null,
      objetivos_completados: [],
      logros_desbloqueados: [],
      eventos: [],
    };
  }

  const data = await requestJson(`/api/gamificacion/sincronizar/${encodeURIComponent(id)}`, {
    method: "POST",
  });

  return {
    ...data,
    eventos: data.eventos ?? [],
  };
}

export async function obtenerLogrosUsuario(usuarioId) {
  const id = normalizeUserId(usuarioId);

  if (!id) {
    return {
      total: 0,
      desbloqueados: 0,
      pendientes: 0,
      logros: [],
    };
  }

  return requestJson(`/api/logros-usuario/${encodeURIComponent(id)}`);
}

export async function obtenerContenidoAprendizaje() {
  return requestJson("/api/contenido-aprendizaje");
}

export async function registrarRondaMinijuego(payload) {
  const usuarioId = normalizeUserId(payload?.usuario_id);

  if (!usuarioId) {
    return { omitido: true, motivo: "usuario_no_registrado", eventos: [] };
  }

  return requestJson("/api/rondas-minijuego", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      usuario_id: usuarioId,
      categoria_id: payload.categoria_id,
      cantidad_minijuegos: payload.cantidad_minijuegos,
      correctas: payload.correctas,
    }),
  });
}

export async function obtenerMarcos(usuarioId = null) {
  const id = normalizeUserId(usuarioId);
  const query = id ? `?usuario_id=${encodeURIComponent(id)}` : "";
  return requestJson(`/api/marcos${query}`);
}

export async function obtenerTitulos(usuarioId = null) {
  const id = normalizeUserId(usuarioId);
  const query = id ? `?usuario_id=${encodeURIComponent(id)}` : "";
  return requestJson(`/api/titulos${query}`);
}

export async function equiparPerfil(payload) {
  return requestJson("/api/perfil/equipamiento", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      usuario_id: normalizeUserId(payload.usuario_id),
      marco_id: payload.marco_id,
      titulo_id: payload.titulo_id,
    }),
  });
}
