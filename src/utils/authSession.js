const AUTH_TOKEN_KEY = "lsa_auth_token";
const AUTH_USER_KEY = "lsa_auth_user";
const AUTH_USER_ID_KEY = "lsa_usuario_registrado_id";

function safeParse(value) {
  try {
    return value ? JSON.parse(value) : null;
  } catch {
    return null;
  }
}

export function getAuthToken() {
  return window.localStorage.getItem(AUTH_TOKEN_KEY);
}

export function getAuthUser() {
  return safeParse(window.localStorage.getItem(AUTH_USER_KEY));
}

export function getAuthUserId() {
  const user = getAuthUser();
  const fromUser = Number(user?.id_usuario ?? user?.id);
  if (Number.isFinite(fromUser) && fromUser > 0) return fromUser;

  const stored = Number(window.localStorage.getItem(AUTH_USER_ID_KEY));
  return Number.isFinite(stored) && stored > 0 ? stored : null;
}

export function setAuthSession({ token, usuario }) {
  if (token) {
    window.localStorage.setItem(AUTH_TOKEN_KEY, token);
  }

  if (usuario) {
    window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(usuario));
    const id = Number(usuario.id_usuario ?? usuario.id);
    if (Number.isFinite(id) && id > 0) {
      window.localStorage.setItem(AUTH_USER_ID_KEY, String(id));
    }
  }
}

export function clearAuthSession() {
  window.localStorage.removeItem(AUTH_TOKEN_KEY);
  window.localStorage.removeItem(AUTH_USER_KEY);
  window.localStorage.removeItem(AUTH_USER_ID_KEY);
}
