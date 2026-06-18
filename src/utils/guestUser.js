import { clearAuthSession, getAuthUserId } from "./authSession";

export function getRegisteredUserId() {
  return getAuthUserId();
}

export function setRegisteredUserId(id) {
  const value = Number(id);
  if (Number.isFinite(value) && value > 0) {
    window.localStorage.setItem("lsa_usuario_registrado_id", String(value));
  }
}

export function clearRegisteredUserId() {
  clearAuthSession();
}
