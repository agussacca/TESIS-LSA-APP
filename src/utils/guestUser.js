const REGISTERED_USER_ID_STORAGE = "lsa_usuario_registrado_id";

export function getRegisteredUserId() {
  const raw = window.localStorage.getItem(REGISTERED_USER_ID_STORAGE);
  const value = Number(raw);

  return Number.isFinite(value) && value > 0 ? value : null;
}

export function setRegisteredUserId(id) {
  const value = Number(id);

  if (!Number.isFinite(value) || value <= 0) {
    window.localStorage.removeItem(REGISTERED_USER_ID_STORAGE);
    return null;
  }

  window.localStorage.setItem(REGISTERED_USER_ID_STORAGE, String(value));
  return value;
}

export function clearRegisteredUserId() {
  window.localStorage.removeItem(REGISTERED_USER_ID_STORAGE);
}

// Compatibilidad con versiones previas: el invitado ya no se persiste.
export function getOrCreateGuestUserKey() {
  return null;
}
