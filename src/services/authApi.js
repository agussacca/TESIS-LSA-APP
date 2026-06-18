import { clearAuthSession, getAuthToken, setAuthSession } from "../utils/authSession";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function buildUrl(path) {
  return `${API_BASE_URL}${path}`;
}

function normalizeApiErrorMessage(detail, fallback) {
  if (!detail) return fallback;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item?.msg) return String(item.msg).replace(/^Value error,\s*/i, "");
        return null;
      })
      .filter(Boolean);

    if (messages.length > 0) {
      return messages.join(" ");
    }
  }

  if (typeof detail === "object" && detail.msg) {
    return String(detail.msg).replace(/^Value error,\s*/i, "");
  }

  return fallback;
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
    let message = `Error HTTP ${response.status}`;

    try {
      const parsed = JSON.parse(text);
      message = normalizeApiErrorMessage(parsed.detail, message);
    } catch {
      if (text) message = text;
    }

    const error = new Error(message);
    error.status = response.status;
    error.body = text;
    throw error;
  }

  return response.json();
}

function persistAuthResponse(data) {
  setAuthSession({
    token: data.access_token,
    usuario: data.usuario,
  });
  return data;
}

export async function registrarUsuario(payload) {
  const data = await requestJson("/api/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return persistAuthResponse(data);
}

export async function iniciarSesion(payload) {
  const data = await requestJson("/api/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return persistAuthResponse(data);
}

export async function obtenerSesionActual() {
  return requestJson("/api/auth/me");
}


export async function actualizarPerfilUsuario(payload) {
  const data = await requestJson("/api/auth/me", {
    method: "PATCH",
    body: JSON.stringify(payload),
  });

  if (data?.usuario) {
    setAuthSession({
      token: data.access_token || getAuthToken(),
      usuario: data.usuario,
    });
  }

  return data;
}

export async function cerrarSesion() {
  try {
    await requestJson("/api/auth/logout", { method: "POST" });
  } catch {
    // El cierre real del lado cliente es eliminar el token.
  } finally {
    clearAuthSession();
  }
}
