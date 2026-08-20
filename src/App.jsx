/*App.jsx*/
// APP_PROFILE_HOME_V18_MAP_VIEWBOX_FIXED
// PROFILE_HOME_MAP_V19: mapa estable con viewBox preprocesado, host absoluto y slots sobre label_points.
import { useEffect, useMemo, useRef, useState } from "react";
import BackendDiagnostics from "./components/dev/BackendDiagnostics";
import CameraView from "./components/camera/CameraView";
import "./App.css";
import "./auth.css";

import { getHealth } from "./services/apiClient";
import { useRecognitionSocket } from "./hooks/useRecognitionSocket";
import CameraFrameDiagnostics from "./components/dev/CameraFrameDiagnostics";
import EvaluatePracticePanel from "./components/practice/EvaluatePracticePanel";
import EvaluateDiagnostics from "./components/dev/EvaluateDiagnostics";
import PracticeProgressPanel from "./components/progress/PracticeProgressPanel";
import UserPanelSummary from "./components/progress/UserPanelSummary";
import { clearAuthSession, getAuthToken, getAuthUserId, setAuthSession } from "./utils/authSession";
import {
  obtenerUsuarioRegistrado,
  obtenerPanelUsuario,
  obtenerObjetivosUsuario,
  sincronizarGamificacionUsuario,
  obtenerLogrosUsuario,
  obtenerContenidoAprendizaje,
  registrarRondaMinijuego,
  obtenerMarcos,
  obtenerTitulos,
  equiparPerfil,
} from "./services/progressApi";
import { actualizarPerfilUsuario, cerrarSesion, iniciarSesion, obtenerSesionActual, registrarUsuario } from "./services/authApi";
import GuidedSpellPanel from "./components/practice/GuidedSpellPanel";
import FreeSpellPanel from "./components/practice/FreeSpellPanel";

const DEFAULT_PROFILE_DATA = {
  name: "Usuario",
  email: "",
  phone: "+54 9 387 000-0000",
  photo: "J",
  title: "Aprendiz constante",
  titleDbId: null,
  frame: null,
  frameDbId: null,
  frameName: "",
  frameImageUrl: null,
};

function normalizeFrameOptionFromApi(item) {
  return {
    id: `marco-${item.id_marco ?? item.id}`,
    dbId: item.id_marco ?? item.id,
    name: item.nombre,
    level: item.nivel_requerido ? `Nivel ${item.nivel_requerido}` : "Especial",
    imageUrl: item.imagen_url,
    disponible: item.disponible !== false,
    order: item.orden ?? 0,
  };
}

function normalizeTitleOptionFromApi(item) {
  return {
    id: `titulo-${item.id_titulo ?? item.id}`,
    dbId: item.id_titulo ?? item.id,
    name: item.nombre,
    level: `Nivel ${item.nivel_requerido ?? 1}`,
    disponible: item.disponible !== false,
    order: item.orden ?? 0,
  };
}

function frameFromProfileData(profileData) {
  if (!profileData?.frameImageUrl) return null;

  return {
    id: profileData.frame,
    dbId: profileData.frameDbId,
    name: profileData.frameName || "Marco de perfil",
    imageUrl: profileData.frameImageUrl,
  };
}

const PROFILE_PHOTOS = [
  { id: "j", label: "J" },
  { id: "🙂", label: "🙂" },
  { id: "😎", label: "😎" },
  { id: "🧑", label: "🧑" },
  { id: "👩", label: "👩" },
  { id: "👨", label: "👨" },
];

export default function App() {
  const [screen, setScreen] = useState("access");
  const [isGuest, setIsGuest] = useState(false);
  const [usuarioId, setUsuarioId] = useState(() => getAuthUserId());
  const [activeCategory, setActiveCategory] = useState(null);
  const [categoryModal, setCategoryModal] = useState(null);
  const [selectedPreview, setSelectedPreview] = useState(null);
  const [detailsReturnScreen, setDetailsReturnScreen] = useState("home");
  const [practiceInitialLetter, setPracticeInitialLetter] = useState("A");
  const [practiceSingleSign, setPracticeSingleSign] = useState(false);
  const [profileData, setProfileData] = useState(DEFAULT_PROFILE_DATA);
  const [userProgress, setUserProgress] = useState(() => normalizeGamificationProgress(null));
  const [learningCategories, setLearningCategories] = useState([]);
  const [signsByCategory, setSignsByCategory] = useState({});
  const [learningContentStatus, setLearningContentStatus] = useState("loading");

  const unlockedCategories = useMemo(() => {
    if (isGuest) {
      return learningCategories
        .filter((category) => category.guestAvailable)
        .map((category) => category.id);
    }

    return learningCategories.map((category) => category.id);
  }, [isGuest, learningCategories]);

  const [backendStatus, setBackendStatus] = useState(null);
  const [socketEnabled, setSocketEnabled] = useState(false);

  const [gamificationQueue, setGamificationQueue] = useState([]);
  const [gamificationToast, setGamificationToast] = useState(null);

  const {
    connected,
    lastMessage,
    error: socketError,
    sendTestFrame,
  } = useRecognitionSocket(socketEnabled);

  useEffect(() => {
    getHealth()
      .then(setBackendStatus)
      .catch((error) => setBackendStatus({ status: "error", message: error.message }));
  }, []);


  useEffect(() => {
    let cancelled = false;

    async function restoreAuthenticatedSession() {
      if (!getAuthToken()) return;

      try {
        const data = await obtenerSesionActual();
        if (cancelled) return;

        const usuario = data.usuario;
        const id = Number(usuario?.id_usuario ?? usuario?.id);
        if (!Number.isFinite(id) || id <= 0) {
          clearAuthSession();
          return;
        }

        setAuthSession({ token: getAuthToken(), usuario });
        setIsGuest(false);
        setUsuarioId(id);
        await refreshUserData(id);
        if (!cancelled) {
          setScreen("home");
        }
      } catch (error) {
        console.warn("No se pudo restaurar la sesión:", error);
        clearAuthSession();
        if (!cancelled) {
          setUsuarioId(null);
        }
      }
    }

    restoreAuthenticatedSession();

    return () => {
      cancelled = true;
    };
  }, []);

  async function enterAsUser(credentials) {
    const data = await iniciarSesion(credentials);
    const usuario = data.usuario;
    const id = Number(usuario?.id_usuario ?? usuario?.id);

    if (!Number.isFinite(id) || id <= 0) {
      throw new Error("No se pudo identificar al usuario autenticado.");
    }

    setAuthSession({ token: data.access_token, usuario });
    setIsGuest(false);
    setUsuarioId(id);
    await refreshUserData(id);
    setScreen("home");
  }

  async function registerAsUser(payload) {
    const data = await registrarUsuario(payload);
    const usuario = data.usuario;
    const id = Number(usuario?.id_usuario ?? usuario?.id);

    if (!Number.isFinite(id) || id <= 0) {
      throw new Error("No se pudo identificar al usuario registrado.");
    }

    setAuthSession({ token: data.access_token, usuario });
    setIsGuest(false);
    setUsuarioId(id);
    await refreshUserData(id);
    setScreen("home");
  }


  async function updateAuthenticatedProfile(patch) {
    const data = await actualizarPerfilUsuario(patch);
    const usuario = data?.usuario;

    if (!usuario) {
      return null;
    }

    setProfileData((prev) => ({
      ...prev,
      name: usuario.nombre_visible || prev.name,
      email: usuario.email || prev.email,
      photo: usuario.foto_perfil_url || prev.photo,
    }));

    return usuario;
  }

  function enterAsGuest() {
    clearAuthSession();
    setUsuarioId(null);
    setIsGuest(true);
    setUserProgress(normalizeGamificationProgress(null));
    setScreen("home");
  }

  async function logout() {
    await cerrarSesion();
    clearAuthSession();
    setUsuarioId(null);
    setIsGuest(false);
    setProfileData(DEFAULT_PROFILE_DATA);
    setUserProgress(normalizeGamificationProgress(null));
    setScreen("access");
  }

  function openCategory(category) {
    if (!unlockedCategories.includes(category.id)) return;
    setCategoryModal(category);
  }

  function goToCategorySection(category, section) {
    setActiveCategory(category);
    setCategoryModal(null);

    if (section === "camera") {
      setPracticeInitialLetter("A");
      setPracticeSingleSign(false);
    }

    setScreen(section);
  }

  function goHome() {
    setScreen("home");
    setActiveCategory(null);
  }

  async function refreshUserData(id = usuarioId) {
    if (!id || isGuest) {
      setUserProgress(normalizeGamificationProgress(null));
      return;
    }

    try {
      const [usuarioResult, panelResult, marcosResult, titulosResult] = await Promise.allSettled([
        obtenerUsuarioRegistrado(id),
        obtenerPanelUsuario(id),
        obtenerMarcos(id),
        obtenerTitulos(id),
      ]);

      const panel = panelResult.status === "fulfilled" ? panelResult.value : null;
      const usuario = usuarioResult.status === "fulfilled" ? usuarioResult.value : panel?.usuario ?? null;
      const marcos = marcosResult.status === "fulfilled" && Array.isArray(marcosResult.value)
        ? marcosResult.value.map(normalizeFrameOptionFromApi)
        : [];
      const titulos = titulosResult.status === "fulfilled" && Array.isArray(titulosResult.value)
        ? titulosResult.value.map(normalizeTitleOptionFromApi)
        : [];

      if (panel?.progreso) {
        setUserProgress(normalizeGamificationProgress(panel.progreso));
      }

      const marcoEquipadoId = usuario?.marco_equipado_id ?? usuario?.marco_equipado?.id_marco ?? null;
      const tituloEquipadoId = usuario?.titulo_equipado_id ?? usuario?.titulo_equipado?.id_titulo ?? null;

      const marcoEquipado = usuario?.marco_equipado
        ? normalizeFrameOptionFromApi(usuario.marco_equipado)
        : marcos.find((marco) => marco.dbId === marcoEquipadoId) || marcos.find((marco) => marco.disponible) || null;

      const tituloEquipado = usuario?.titulo_equipado
        ? normalizeTitleOptionFromApi(usuario.titulo_equipado)
        : titulos.find((titulo) => titulo.dbId === tituloEquipadoId) || titulos.find((titulo) => titulo.disponible) || null;

      setProfileData((prev) => ({
        ...prev,
        name: usuario?.nombre_visible || prev.name,
        email: usuario?.email || prev.email,
        photo: usuario?.foto_perfil_url || prev.photo,
        frame: marcoEquipado?.id ?? prev.frame,
        frameDbId: marcoEquipado?.dbId ?? prev.frameDbId,
        frameName: marcoEquipado?.name ?? prev.frameName,
        frameImageUrl: marcoEquipado?.imageUrl ?? prev.frameImageUrl,
        title: tituloEquipado?.name ?? prev.title,
        titleDbId: tituloEquipado?.dbId ?? prev.titleDbId,
      }));
    } catch (error) {
      console.warn("No se pudo cargar el perfil desde la base de datos:", error);
    }
  }

  useEffect(() => {
    if (!isGuest && usuarioId) {
      refreshUserData(usuarioId);
      return;
    }

    if (isGuest) {
      setUserProgress(normalizeGamificationProgress(null));
    }
  }, [isGuest, usuarioId]);

  function pushGamificationEvents(events) {
    if (!Array.isArray(events) || events.length === 0) return;

    const timestamp = Date.now();

    const normalizedEvents = events.map((event, index) => ({
      ...event,
      __toastKey: `${timestamp}-${index}-${event.tipo}-${event.objetivo_id ?? event.logro_id ?? event.nivel_nuevo ?? "general"}`,
    }));

    setGamificationQueue((prev) => [...prev, ...normalizedEvents]);
  }

  async function syncGamificationNow() {
    if (isGuest || !usuarioId) {
      return null;
    }

    try {
      const data = await sincronizarGamificacionUsuario(usuarioId);

      if (data?.progreso) {
        setUserProgress(normalizeGamificationProgress(data.progreso));
      }

      if (Array.isArray(data.eventos) && data.eventos.length > 0) {
        pushGamificationEvents(data.eventos);
      }

      await refreshUserData(usuarioId);

      return data;
    } catch (error) {
      console.error("No se pudo sincronizar gamificación:", error);
      return null;
    }
  }

  useEffect(() => {
    if (gamificationToast || gamificationQueue.length === 0) return;

    const [next, ...rest] = gamificationQueue;

    setGamificationToast(next);
    setGamificationQueue(rest);
  }, [gamificationQueue, gamificationToast]);

  useEffect(() => {
    if (!gamificationToast) return;

    const duration =
      gamificationToast.tipo === "nivel"
        ? 4600
        : gamificationToast.tipo === "logro"
          ? 4600
          : gamificationToast.tipo === "objetivo"
            ? 3400
            : 2600;

    const timeout = setTimeout(() => {
      setGamificationToast(null);
    }, duration);

    return () => clearTimeout(timeout);
  }, [gamificationToast]);

  useEffect(() => {
    let cancelled = false;

    async function loadLearningContent() {
      setLearningContentStatus("loading");

      try {
        const data = await obtenerContenidoAprendizaje();

        if (cancelled) return;

        const normalized = normalizeLearningContent(data);

        setLearningCategories(normalized.categories);
        setSignsByCategory(normalized.signsByCategory);
        setLearningContentStatus("ready");
      } catch (error) {
        console.error("No se pudo cargar el contenido de aprendizaje:", error);

        if (!cancelled) {
          setLearningCategories([]);
          setSignsByCategory({});
          setLearningContentStatus("error");
        }
      }
    }

    loadLearningContent();

    return () => {
      cancelled = true;
    };
  }, []);

  if (screen === "access") {
    return (
      <AccessScreen
        onLogin={() => setScreen("login")}
        onRegister={() => setScreen("register")}
        onGuest={enterAsGuest}
      />
    );
  }

  if (screen === "login") {
    return (
      <AuthScreen
        mode="login"
        onSubmit={enterAsUser}
        onBack={() => setScreen("access")}
        onGuest={enterAsGuest}
        onSwitchMode={() => setScreen("register")}
      />
    );
  }

  if (screen === "register") {
    return (
      <AuthScreen
        mode="register"
        onSubmit={registerAsUser}
        onBack={() => setScreen("access")}
        onGuest={enterAsGuest}
        onSwitchMode={() => setScreen("login")}
      />
    );
  }

  return (
    <div className="app">
      <Header
        isGuest={isGuest}
        profileData={profileData}
        userProgress={userProgress}
        onHome={goHome}
        onProfile={() => setScreen("profile")}
        onStats={() => {
          setDetailsReturnScreen("home");
          setScreen("stats");
        }}
        onLogout={logout}
      />

      {/*<BackendDiagnostics />
      {/*<CameraFrameDiagnostics />
      <EvaluateDiagnostics />*/}

      <main className="main page-in">
        {screen === "home" && (
          <HomeScreen
            isGuest={isGuest}
            profileData={profileData}
            usuarioId={usuarioId}
            categories={learningCategories}
            contentStatus={learningContentStatus}
            unlockedCategories={unlockedCategories}
            onCategoryClick={openCategory}
            onProfile={() => setScreen("profile")}
            onStats={() => { setDetailsReturnScreen("home"); setScreen("stats"); }}
            onAchievements={() => { setDetailsReturnScreen("home"); setScreen("achievements"); }}
          />
        )}

        {screen === "learn" && activeCategory && (
          <LearnScreen
            category={activeCategory}
            signs={signsByCategory[activeCategory.id] || []}
            onBack={goHome}
            onOpenPreview={setSelectedPreview}
          />
        )}

        {screen === "camera" && (
          <CameraPracticeScreen
            onBack={goHome}
            initialLetter={practiceInitialLetter}
            singleSignMode={practiceSingleSign}
            signs={signsByCategory.abecedario || []}
            onGamificationSync={syncGamificationNow}
            usuarioId={usuarioId}
            persistEnabled={!isGuest}
          />
        )}

        {screen === "spell" && (
          <SpellScreen
            onBack={goHome}
            onGamificationSync={syncGamificationNow}
            usuarioId={usuarioId}
            persistEnabled={!isGuest}
          />
        )}

        {screen === "challenges" && activeCategory && (
          <ChallengesScreen
            category={activeCategory}
            onBack={goHome}
            onOpenPreview={setSelectedPreview}
            onGamificationSync={syncGamificationNow}
            onGamificationEvents={pushGamificationEvents}
            usuarioId={usuarioId}
            persistEnabled={!isGuest}
          />
        )}

        {screen === "profile" && (
          <ProfileScreen
            isGuest={isGuest}
            profileData={profileData}
            setProfileData={setProfileData}
            onUpdateProfile={updateAuthenticatedProfile}
            usuarioId={usuarioId}
            userProgress={userProgress}
            onBack={goHome}
            onStats={() => { setDetailsReturnScreen("profile"); setScreen("stats"); }}
            onAchievements={() => { setDetailsReturnScreen("profile"); setScreen("achievements"); }}
          />
        )}

        {screen === "stats" && (
          <StatsScreen
            onBack={() => setScreen(detailsReturnScreen)}
            usuarioId={usuarioId}
            isGuest={isGuest}
          />
        )}

        {screen === "achievements" && (
          <AchievementsScreen
            onBack={() => setScreen(detailsReturnScreen)}
            usuarioId={usuarioId}
            isGuest={isGuest}
          />
        )}
      </main>

      {categoryModal && (
        <CategoryModal
          category={categoryModal}
          onClose={() => setCategoryModal(null)}
          onGo={goToCategorySection}
        />
      )}

      {selectedPreview && (
        <CardPreviewModal
          preview={selectedPreview}
          onClose={() => setSelectedPreview(null)}
          onPracticeSign={
            activeCategory?.id === "abecedario"
              ? (sign) => {
                  setPracticeInitialLetter(sign.name);
                  setPracticeSingleSign(true);
                  setSelectedPreview(null);
                  setScreen("camera");
                }
              : null
          }
        />
      )}
      {gamificationToast && (
        <GamificationToast
          key={gamificationToast.__toastKey}
          event={gamificationToast}
          onClose={() => setGamificationToast(null)}
        />
      )}
    </div>
  );
}

function AccessScreen({ onLogin, onRegister, onGuest }) {
  return (
    <div className="access-screen access-landing-screen">
      <section className="access-content fade-up">
        <div className="tag">Aprendizaje inicial de LSA</div>
        <h1>Aprendé señas con videos, cámara y desafíos.</h1>
        <p>
          Una web app gamificada para practicar el Abecedario, aprender vocabulario básico
          y reforzar lo aprendido con minijuegos.
        </p>

        <div className="access-buttons">
          <button type="button" className="primary" onClick={onLogin}>
            Iniciar sesión
          </button>
          <button type="button" className="secondary" onClick={onRegister}>
            Registrarme
          </button>
          <button type="button" className="ghost" onClick={onGuest}>
            Ingresar como invitado
          </button>
        </div>

        <small>
          Como invitado podés probar el Abecedario sin guardar progreso. Iniciá sesión para
          conservar tu avance y acceder a la experiencia completa.
        </small>
      </section>

      <section className="access-card fade-up delay-1">
        <Feature title="Práctica con cámara" text="Realizá señas del Abecedario y recibí una devolución." icon="📷" />
        <Feature title="Tarjetas de aprendizaje" text="Consultá descripción, imagen y video demostrativo." icon="🃏" />
        <Feature title="Desafíos interactivos" text="Completá rondas con selección, arrastre, frases y mapas." icon="🎮" />
      </section>
    </div>
  );
}

function AuthScreen({ mode, onSubmit, onBack, onGuest, onSwitchMode }) {
  const isRegister = mode === "register";
  const [nombreVisible, setNombreVisible] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setErrorMessage("");

    const normalizedEmail = email.trim().toLowerCase();
    const normalizedName = nombreVisible.trim();

    if (isRegister && !normalizedName) {
      setErrorMessage("Ingresá tu nombre.");
      return;
    }

    if (!normalizedEmail) {
      setErrorMessage("Ingresá tu correo electrónico.");
      return;
    }

    if (!password) {
      setErrorMessage("Ingresá tu contraseña.");
      return;
    }

    if (isRegister && password.length < 8) {
      setErrorMessage("La contraseña debe tener al menos 8 caracteres.");
      return;
    }

    if (isRegister && !confirmPassword) {
      setErrorMessage("Confirmá tu contraseña.");
      return;
    }

    if (isRegister && password !== confirmPassword) {
      setErrorMessage("Las contraseñas no coinciden.");
      return;
    }

    setSubmitting(true);

    try {
      if (isRegister) {
        await onSubmit({
          email: normalizedEmail,
          password,
          nombre_visible: normalizedName,
          foto_perfil_url: null,
        });
      } else {
        await onSubmit({ email: normalizedEmail, password });
      }
    } catch (error) {
      setErrorMessage(error.message || "No se pudo completar la operación.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-standalone-screen fade-up">
      <section className="auth-standalone-card">
        <div className="auth-standalone-icon" aria-hidden="true">
          {isRegister ? "👥" : "🔐"}
        </div>

        <div className="auth-standalone-copy">
          <h1>{isRegister ? "Crear cuenta" : "Iniciar sesión"}</h1>
          <p>
            {isRegister
              ? "Completá tus datos para guardar el avance en la plataforma."
              : "Ingresá con tu cuenta para recuperar tu progreso, logros y personalización."}
          </p>
        </div>

        <form className="auth-standalone-form" onSubmit={handleSubmit} noValidate>
          {isRegister && (
            <label className="auth-field">
              <span>Nombre</span>
              <input
                value={nombreVisible}
                onChange={(event) => setNombreVisible(event.target.value)}
                placeholder="Tu nombre"
                autoComplete="name"
                disabled={submitting}
              />
            </label>
          )}

          <label className="auth-field">
            <span>Correo electrónico</span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="usuario@mail.com"
              autoComplete="email"
              disabled={submitting}
            />
          </label>

          <label className="auth-field">
            <span>Contraseña</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={isRegister ? "Mínimo 8 caracteres" : "Tu contraseña"}
              autoComplete={isRegister ? "new-password" : "current-password"}
              disabled={submitting}
            />
          </label>

          {isRegister && (
            <label className="auth-field">
              <span>Confirmar contraseña</span>
              <input
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                placeholder="Repetí tu contraseña"
                autoComplete="new-password"
                disabled={submitting}
              />
            </label>
          )}

          {errorMessage && <p className="auth-error">{errorMessage}</p>}

          <button className="primary auth-submit auth-submit-block" type="submit" disabled={submitting}>
            {submitting ? "Procesando..." : isRegister ? "Registrarme" : "Entrar"}
          </button>
        </form>

        <div className="auth-standalone-footer">
          <button type="button" className="auth-inline-link" onClick={onSwitchMode} disabled={submitting}>
            {isRegister ? "Ya tengo cuenta" : "Crear una cuenta"}
          </button>
          <button type="button" className="auth-inline-link auth-back-bottom" onClick={onBack} disabled={submitting}>
            Volver al acceso
          </button>
        </div>
      </section>
    </div>
  );
}

function Feature({ icon, title, text }) {
  return (
    <div className="feature">
      <div className="feature-icon">{icon}</div>
      <div>
        <h3>{title}</h3>
        <p>{text}</p>
      </div>
    </div>
  );
}

function Header({ isGuest, profileData, userProgress, onHome, onProfile, onStats, onLogout }) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="header">
      <button className="brand" onClick={onHome}>
        <span className="brand-logo">LSA</span>
        <span>SeñApp</span>
      </button>

      <div className="profile-area">
        <button className="profile-button" onClick={() => setMenuOpen(!menuOpen)}>
          <div className="header-avatar">
            {isGuest ? (
              "👤"
            ) : isImagePhoto(profileData.photo) ? (
              <img src={profileData.photo} alt="Foto de perfil" />
            ) : (
              profileData.photo
            )}
          </div>

          <div className="profile-summary">
            <strong>{isGuest ? "Invitado" : profileData.name}</strong>
            <span>{isGuest ? "Sin progreso" : `Nivel ${userProgress?.nivel ?? 1}`}</span>
          </div>
        </button>

        {menuOpen && (
          <div className="profile-menu pop-in">
            {isGuest ? (
              <>
                <button onClick={onStats}>Ver estadísticas</button>
                <button onClick={onLogout}>Iniciar sesión</button>
                <button onClick={onLogout}>Registrarme</button>
              </>
            ) : (
              <>
                <button onClick={onProfile}>Ver perfil</button>
                <button onClick={onStats}>Ver estadísticas</button>
              </>
            )}

            <button onClick={onLogout}>Salir</button>
          </div>
        )}
      </div>
    </header>
  );
}

function ProfileFrame({ small = false, isGuest = false }) {
  return (
    <div className={small ? "profile-frame small" : "profile-frame big"}>
      <div className={small ? "profile-photo small" : "profile-photo big"}>
        {isGuest ? "👤" : "J"}
      </div>
    </div>
  );
}

function HomeScreen({
  isGuest,
  profileData,
  usuarioId,
  categories,
  contentStatus,
  unlockedCategories,
  onCategoryClick,
  onProfile,
  onStats,
  onAchievements,
}) {
  const selectedFrame = frameFromProfileData(profileData);

  const [dailyObjectives, setDailyObjectives] = useState([]);
  const [weeklyObjectives, setWeeklyObjectives] = useState([]);
  const [objectivesStatus, setObjectivesStatus] = useState("loading");

  const [gamificationProgress, setGamificationProgress] = useState({
    xpTotal: 0,
    nivel: 1,
    xpNivelActual: 0,
    xpSiguienteNivel: 120,
    rachaActual: 0,
    rachaMaxima: 0,
  });

  useEffect(() => {
    let cancelled = false;

    async function loadObjectives() {
      setObjectivesStatus("loading");

      try {
        if (isGuest || !usuarioId) {
          if (cancelled) return;
          setDailyObjectives([]);
          setWeeklyObjectives([]);
          setGamificationProgress(normalizeGamificationProgress(null));
          setObjectivesStatus("ready");
          return;
        }

        const data = await obtenerObjetivosUsuario(usuarioId);

        if (cancelled) return;

        setDailyObjectives(normalizeObjectives(data.diarios));
        setWeeklyObjectives(normalizeObjectives(data.semanales));
        setGamificationProgress(normalizeGamificationProgress(data.progreso));
        setObjectivesStatus("ready");
      } catch (error) {
        console.error(error);

        if (!cancelled) {
          setObjectivesStatus("error");
          setDailyObjectives([]);
          setWeeklyObjectives([]);
        }
      }
    }

    loadObjectives();

    return () => {
      cancelled = true;
    };
  }, [isGuest, usuarioId]);

  if (isGuest) {
    return (
      <div className="page-stack">
        <section className="guest-home card glass fade-up">
          <div>
            <h2>Modo invitado</h2>
            <p>
              Podés probar el Abecedario sin guardar progreso. Iniciá sesión para
              conservar tu avance, consultar estadísticas y acceder a todas las funciones.
            </p>
          </div>

          <div className="guest-home-actions">
            <button className="primary">Iniciar sesión</button>
            <button className="secondary" onClick={onStats}>
              Ver estadísticas
            </button>
          </div>
        </section>

        <section className="card fade-up delay-1">
          <div className="section-heading compact">
            <div>
              <h3>Categorías de aprendizaje</h3>
              <p>Elegí una categoría para comenzar.</p>
            </div>
          </div>

          <div className="category-grid">

            {contentStatus === "loading" && (
              <p className="empty-objectives-message">
                Cargando categorías...
              </p>
            )}

            {contentStatus === "error" && (
              <p className="empty-objectives-message">
                No se pudieron cargar las categorías.
              </p>
            )}

            {contentStatus === "ready" &&
              categories.map((category) => (
                <CategoryCard
                  key={category.id}
                  category={category}
                  locked={!unlockedCategories.includes(category.id)}
                  onClick={() => onCategoryClick(category)}
                />
              ))}

          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="home-dashboard-v9 fade-up">
      <aside className="home-profile-panel card">
        <IdentityFrame
          photo={profileData.photo}
          frame={selectedFrame}
          size="large"
        />

        <div className="home-profile-copy">
          <h2>{profileData.name}</h2>
          <p>{profileData.title}</p>
        </div>

        <div className="home-player-level">
          <strong>Nivel {gamificationProgress.nivel}</strong>
          <span>
            {gamificationProgress.xpNivelActual}/{gamificationProgress.xpSiguienteNivel} XP
          </span>
          <div className="progress slim">
            <div
              style={{
                width: `${Math.min(
                  (gamificationProgress.xpNivelActual /
                    Math.max(gamificationProgress.xpSiguienteNivel, 1)) *
                    100,
                  100
                )}%`,
              }}
            />
          </div>
        </div>

        <div className="home-streak-card">
          <span>🔥</span>
          <div>
            <strong>{gamificationProgress.rachaActual} días</strong>
            <small>Racha actual</small>
          </div>
        </div>

        <button className="primary" onClick={onProfile}>
          Ver perfil
        </button>

        <button className="secondary" onClick={onStats}>
          Ver estadísticas
        </button>
      </aside>

      <section className="home-category-panel card">
        <div className="section-heading compact">
          <div>
            <h3>Categorías de aprendizaje</h3>
            <p>Elegí una categoría para aprender, practicar con cámara o jugar desafíos.</p>
          </div>
        </div>

        <div className="category-grid">
          {contentStatus === "loading" && (
            <p className="empty-objectives-message">
              Cargando categorías...
            </p>
          )}

          {contentStatus === "error" && (
            <p className="empty-objectives-message">
              No se pudieron cargar las categorías.
            </p>
          )}

          {contentStatus === "ready" &&
            categories.map((category) => (
              <CategoryCard
                key={category.id}
                category={category}
                locked={!unlockedCategories.includes(category.id)}
                onClick={() => onCategoryClick(category)}
              />
            ))}
        </div>

      </section>

      <aside className="home-objectives-panel">
        <div className="home-objective-card card">
          <div className="home-panel-header">
            <div>
              <h3>Objetivos diarios</h3>
              <p>Metas rápidas para mantener la práctica.</p>
              {objectivesStatus === "loading" && (
                <small className="objectives-status">Actualizando objetivos...</small>
              )}

              {objectivesStatus === "error" && (
                <small className="objectives-status error">
                  No se pudieron cargar los objetivos reales.
                </small>
              )}
            </div>
          </div>

          <div className="weekly-objective-list">
            {dailyObjectives.length === 0 ? (
              <p className="empty-objectives-message">
                {objectivesStatus === "loading"
                  ? "Cargando objetivos..."
                  : "No hay objetivos diarios activos."}
              </p>
            ) : (
              dailyObjectives.map((objective) => (
                <WeeklyObjectiveCard
                  key={objective.codigo || objective.id}
                  objective={objective}
                />
              ))
            )}
          </div>

        </div>

        <div className="home-objective-card card">
          <div className="home-panel-header">
            <div>
              <h3>Objetivos semanales</h3>
              <p>Dan más experiencia.</p>
            </div>
            <span className="weekly-days">3 días</span>
          </div>

          <div className="weekly-objective-list">
            {weeklyObjectives.length === 0 ? (
              <p className="empty-objectives-message">
                {objectivesStatus === "loading"
                  ? "Cargando objetivos..."
                  : "No hay objetivos semanales activos."}
              </p>
            ) : (
              weeklyObjectives.map((objective) => (
                <WeeklyObjectiveCard
                  key={objective.codigo || objective.id}
                  objective={objective}
                />
              ))
            )}
          </div>

        </div>
      </aside>
    </div>
  );
}

function normalizeObjectives(items) {
  if (!Array.isArray(items)) {
    return [];
  }

  return items.map((item) => ({
    id: item.id,
    codigo: item.codigo,
    title: item.titulo ?? item.title,
    current: item.actual ?? item.current ?? 0,
    total: item.objetivo ?? item.total ?? 1,
    xp: item.xp ?? 0,
    completed: Boolean(item.completado ?? item.completed),
    xpAwarded: Boolean(item.xp_otorgado ?? item.xpAwarded),
  }));
}

function normalizeAchievements(items) {
  if (!Array.isArray(items)) {
    return [];
  }

  return items.map((item) => ({
    id: item.id_logro ?? item.id,
    codigo: item.codigo ?? item.id_logro,
    family: item.familia,
    name: item.nombre,
    description: item.descripcion,
    imageUrl: item.imagen_url,
    unlocked: Boolean(item.desbloqueado ?? item.obtenido),
    unlockedAt: item.fecha_desbloqueo,
    order: item.orden ?? item.id_logro ?? 0,
  }));
}

function slugifyCategoryName(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/ñ/g, "n")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function normalizeLearningContent(items) {
  if (!Array.isArray(items)) {
    return {
      categories: [],
      signsByCategory: {},
    };
  }

  const categories = items.map(normalizeLearningCategory);

  const signsByCategory = Object.fromEntries(
    items.map((category) => {
      const normalizedCategory = normalizeLearningCategory(category);
      return [
        normalizedCategory.id,
        Array.isArray(category.senias)
          ? category.senias.map((sign) => normalizeLearningSign(sign, normalizedCategory))
          : [],
      ];
    })
  );

  return {
    categories,
    signsByCategory,
  };
}


function normalizeLearningCategory(item) {
  const name = item.nombre ?? item.name ?? "";
  const slug = item.codigo ?? slugifyCategoryName(name);

  return {
    id: slug,
    dbId: item.id_categoria_aprendizaje ?? item.id,
    codigo: slug,
    name,
    icon: item.icono ?? "📚",
    color: item.color,
    description: item.descripcion,
    imageUrl: item.imagen_portada_url,
    order: item.orden ?? 0,
    active: true,
    guestAvailable: slug === "abecedario",
  };
}


function normalizeLearningSign(item, category = null) {
  const name = item.nombre ?? item.name ?? "";

  return {
    id: item.id_senia ?? item.id ?? name,
    dbId: item.id_senia ?? item.id,
    codigo: item.codigo ?? slugifyCategoryName(name),
    categoryId: category?.id ?? item.categoria_codigo,
    name,
    thumb: item.imagen_url,
    description: item.descripcion,
    imageUrl: item.imagen_url,
    videoUrl: item.video_url,
    order: item.orden ?? 0,
    active: true,
    cameraPractice: category?.id === "abecedario",
  };
}

function groupAchievementsByFamily(items) {
  return items.reduce((groups, achievement) => {
    const family = achievement.family || "General";

    if (!groups[family]) {
      groups[family] = [];
    }

    groups[family].push(achievement);

    return groups;
  }, {});
}

function normalizeGamificationProgress(progreso) {
  if (!progreso) {
    return {
      xpTotal: 0,
      nivel: 1,
      xpNivelActual: 0,
      xpSiguienteNivel: 120,
      rachaActual: 0,
      rachaMaxima: 0,
    };
  }

  return {
    xpTotal: progreso.xp_total ?? 0,
    nivel: progreso.nivel ?? 1,
    xpNivelActual: progreso.xp_nivel_actual ?? 0,
    xpSiguienteNivel: progreso.xp_siguiente_nivel ?? 120,
    rachaActual: progreso.racha_actual ?? 0,
    rachaMaxima: progreso.racha_maxima ?? 0,
  };
}

function isImagePhoto(photo) {
  return typeof photo === "string" && (photo.startsWith("data:image") || photo.startsWith("/") || photo.startsWith("http"));
}

function IdentityFrame({ photo, frame, size = "normal", editable = false, onEdit }) {
  const frameImage = frame?.imageUrl;

  return (
    <div className={`identity-frame-png ${size}`}>
      <div className="identity-photo-mask">
        {isImagePhoto(photo) ? (
          <img src={photo} alt="Foto de perfil" className="identity-photo-img" />
        ) : (
          <span className="identity-photo-fallback">{photo || "👤"}</span>
        )}
      </div>

      {frameImage ? (
        <img src={frameImage} alt={frame?.name || "Marco de perfil"} className="identity-frame-overlay" />
      ) : (
        <span className="identity-frame-fallback-ring" />
      )}

      {editable && (
        <button className="profile-photo-edit" onClick={onEdit} title="Personalizar identidad">
          ✎
        </button>
      )}
    </div>
  );
}

function WeeklyObjectiveCard({ objective }) {
  const current = Math.min(objective.current, objective.total);
  const percentage = Math.min((current / objective.total) * 100, 100);
  const completed = objective.completed || current >= objective.total;
  const xpAwarded = Boolean(objective.xpAwarded);

  return (
    <div className={`weekly-objective-card ${completed ? "completed" : ""}`}>
      <div className="weekly-objective-title">
        <strong>{objective.title}</strong>
        <span>{current}/{objective.total}</span>
      </div>

      <div className="progress slim">
        <div style={{ width: `${percentage}%` }} />
      </div>

      <small>
        {completed
          ? xpAwarded
            ? "XP obtenido"
            : "Completado"
          : `+${objective.xp} XP`}
      </small>
    </div>
  );
}

function CategoryCard({ category, locked, onClick }) {
  return (
    <button className={`category-card ${locked ? "locked" : ""}`} onClick={onClick}>
      <div className={`category-icon ${category.color}`}>{category.icon}</div>
      <h4>{category.name}</h4>
      <p>Aprendizaje y desafíos</p>

      {locked && (
        <div className="locked-layer">
          <div className="lock-icon">🔒</div>
        </div>
      )}
    </button>
  );
}

function AchievementCard({ achievement }) {
  const [imageError, setImageError] = useState(false);

  return (
    <article
      className={`achievement-card ${
        achievement.unlocked ? "unlocked" : "locked"
      }`}
    >
      <div className="achievement-card-image">
        {achievement.imageUrl && !imageError ? (
          <img
            src={achievement.imageUrl}
            alt={achievement.name}
            onError={() => setImageError(true)}
          />
        ) : (
          <span>{achievement.unlocked ? "🏅" : "🔒"}</span>
        )}
      </div>

      <div className="achievement-card-body">
        <strong>{achievement.name}</strong>
        <p>{achievement.description}</p>
      </div>
    </article>
  );
}

function AchievementsScreen({ onBack, usuarioId, isGuest = false }) {
  const [achievementsData, setAchievementsData] = useState({
    total: 0,
    desbloqueados: 0,
    pendientes: 0,
    logros: [],
  });

  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let cancelled = false;

    async function loadAchievements() {
      setStatus("loading");

      try {
        if (isGuest || !usuarioId) {
          if (cancelled) return;
          setAchievementsData({
            total: 0,
            desbloqueados: 0,
            pendientes: 0,
            logros: [],
          });
          setStatus("ready");
          return;
        }

        const data = await obtenerLogrosUsuario(usuarioId);

        if (cancelled) return;

        const normalizedLogros = normalizeAchievements(data.logros);

        setAchievementsData({
          total: data.total ?? normalizedLogros.length,
          desbloqueados:
            data.desbloqueados ??
            normalizedLogros.filter((achievement) => achievement.unlocked).length,
          pendientes:
            data.pendientes ??
            normalizedLogros.filter((achievement) => !achievement.unlocked).length,
          logros: normalizedLogros,
        });

        setStatus("ready");
      } catch (error) {
        console.error(error);

        if (!cancelled) {
          setStatus("error");
          setAchievementsData({
            total: 0,
            desbloqueados: 0,
            pendientes: 0,
            logros: [],
          });
        }
      }
    }

    loadAchievements();

    return () => {
      cancelled = true;
    };
  }, [isGuest, usuarioId]);

  const groupedAchievements = groupAchievementsByFamily(achievementsData.logros);

  return (
    <div className="achievements-screen page-stack">
      <BackButton onBack={onBack} />

      <section className="card fade-up">
        <div className="section-heading compact">
          <div>
            <h2>Logros</h2>
            <p>
              Las medallas obtenidas aparecen a color. Las pendientes se muestran bloqueadas.
            </p>
          </div>

          <div className="achievement-summary">
            <strong>
              {achievementsData.desbloqueados}/{achievementsData.total}
            </strong>
            <small>desbloqueados</small>
          </div>
        </div>
      </section>

      {status === "loading" && (
        <section className="card fade-up">
          <p className="empty-objectives-message">
            Cargando logros...
          </p>
        </section>
      )}

      {status === "error" && (
        <section className="card fade-up">
          <p className="empty-objectives-message">
            No se pudieron cargar los logros.
          </p>
        </section>
      )}

      {status === "ready" && achievementsData.logros.length === 0 && (
        <section className="card fade-up">
          <p className="empty-objectives-message">
            No hay logros activos.
          </p>
        </section>
      )}

      {status === "ready" && achievementsData.logros.length > 0 && (
        <div className="achievement-family-list">
          {Object.entries(groupedAchievements).map(([family, items]) => (
            <section
              key={family}
              className="card fade-up achievement-family-section"
            >
              <div className="section-heading compact">
                <div>
                  <h3>{family}</h3>
                </div>

                <span className="achievement-family-counter">
                  {items.filter((item) => item.unlocked).length}/{items.length}
                </span>
              </div>

              <div className="achievement-grid">
                {items.map((achievement) => (
                  <AchievementCard
                    key={achievement.codigo || achievement.id}
                    achievement={achievement}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

function CategoryModal({ category, onClose, onGo }) {
  const cameraAvailable = category.id === "abecedario";

  return (
    <div className="modal-overlay">
      <div className="category-modal modal-in">
        <div className="modal-header">
          <div>
            <small>Categoría</small>
            <h2>{category.name}</h2>
          </div>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="modal-options">
          <button onClick={() => onGo(category, "learn")}>
            <span>📚</span>
            <div>
              <strong>Aprendizaje</strong>
              <small>Ver tarjetas, descripciones y videos.</small>
            </div>
          </button>

          <button disabled={!cameraAvailable} onClick={() => onGo(category, "camera")}>
            <span>📷</span>
            <div>
              <strong>Práctica con cámara</strong>
              <small>{cameraAvailable ? "Practicar el Abecedario." : "Disponible en futuras versiones."}</small>
            </div>
          </button>

          {category.id === "abecedario" && (
            <button onClick={() => onGo(category, "spell")}>
              <span>🔤</span>
              <div>
                <strong>Deletrear palabras</strong>
                <small>Practicar palabras letra por letra.</small>
              </div>
            </button>
          )}

          <button onClick={() => onGo(category, "challenges")}>
            <span>🎮</span>
            <div>
              <strong>Desafíos interactivos</strong>
              <small>Completar una ronda de minijuegos.</small>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}

function LearnScreen({ category, signs, onBack, onOpenPreview }) {
  return (
    <div className="page-stack">
      <BackButton onBack={onBack} />

      <div className="screen-top card fade-up">
        <div>
          <h2>Aprendizaje: {category.name}</h2>
          <p className="screen-description">
            Tocá una tarjeta para verla más grande, leer la descripción y reproducir el video demostrativo.
          </p>
        </div>
      </div>

      <div className="sign-grid">
        {signs.map((sign, index) => (
          <SignCard
            key={sign.id}
            sign={sign}
            variant="learning"
            expanded={false}
            largeName={category.id === "abecedario"}
            className={`fade-up delay-${(index % 3) + 1}`}
            onExpand={() =>
              onOpenPreview({
                sign,
                title: sign.name,
                showName: true,
                showDescription: true,
              })
            }
          />
        ))}
      </div>
    </div>
  );
}

function FullscreenIcon() {
  return (
    <span className="icon-fullscreen" aria-hidden="true">
      <span></span><span></span><span></span><span></span>
    </span>
  );
}

function CloseIcon() {
  return <span className="icon-close" aria-hidden="true"></span>;
}

function PlayIcon() {
  return <span className="icon-play" aria-hidden="true"></span>;
}

function MediaSlot({
  sign,
  expanded = false,
  videoRef,
  playing,
  onTogglePlay,
  onVideoEnded,
  onVideoError,
}) {
  const label = sign?.name || sign?.thumb || "?";
  const imageSrc = sign?.imageUrl || sign?.image;
  const videoSrc = sign?.videoUrl || sign?.video;
  const showVideo = expanded && Boolean(videoSrc);

  function handleKeyDown(event) {
    if (!showVideo) return;

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onTogglePlay?.(event);
    }
  }

  return (
    <div
      className={`media-slot ${showVideo ? "has-video" : ""}`}
      onClick={showVideo ? onTogglePlay : undefined}
      onKeyDown={showVideo ? handleKeyDown : undefined}
      role={showVideo ? "button" : undefined}
      tabIndex={showVideo ? 0 : undefined}
      aria-label={
        showVideo
          ? playing
            ? `Pausar video de ${label}`
            : `Reproducir video de ${label}`
          : undefined
      }
      title={showVideo ? (playing ? "Pausar video" : "Reproducir video") : undefined}
    >
      {showVideo ? (
        <video
          ref={videoRef}
          key={videoSrc}
          className="card-video"
          src={videoSrc}
          poster={imageSrc || undefined}
          playsInline
          muted
          preload="metadata"
          controls={false}
          disablePictureInPicture
          controlsList="nodownload noplaybackrate noremoteplayback"
          onEnded={onVideoEnded}
          onError={onVideoError}
        />
      ) : imageSrc ? (
        <img src={imageSrc} alt={`Seña ${label}`} />
      ) : (
        <div className="media-fallback">{sign?.thumb || label}</div>
      )}
    </div>
  );
}

function SignCard({
  sign,
  variant = "learning",
  expanded = false,
  onExpand,
  onCollapse,
  style,
  className = "",
  draggable,
  onDragStart,
  hideToggle = false,
  hideName = false,
  customNameSlot = null,
  showPracticeButton = false,
  emphasizeName = false,
  largeName = false,
  onPractice,
}) {
  const [playing, setPlaying] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const videoRef = useRef(null);

  const isLearning = variant === "learning" || variant === "association";
  const isPracticeModal = expanded && showPracticeButton;

  const showName = isLearning && !expanded && (customNameSlot || !hideName);
  const showExpandedName = isLearning && expanded && !hideName && !isPracticeModal;
  const showDescription = isLearning && expanded && !isPracticeModal;
  const videoSrc = sign?.videoUrl || sign?.video;
  const canPlayVideo = Boolean(videoSrc) && !videoError;
  const expandedTitle =
    sign?.categoryId === "abecedario"
      ? `Letra ${sign?.name || ""}`
      : sign?.name || "Seña";
  const actionLabel = expanded ? "Cerrar" : "Agrandar";

  useEffect(() => {
    setPlaying(false);
    setVideoError(false);
  }, [sign?.id, videoSrc]);
  const actionHandler = expanded ? onCollapse : onExpand;

  function handleAction(event) {
    event.stopPropagation();
    videoRef.current?.pause();
    setPlaying(false);
    actionHandler?.();
  }

  function startVideoPlayback(event) {
    event?.stopPropagation?.();

    if (!canPlayVideo || !videoRef.current) return;

    const playPromise = videoRef.current.play();

    if (playPromise?.then) {
      playPromise
        .then(() => setPlaying(true))
        .catch((error) => {
          console.warn("No se pudo reproducir el video:", error);
          setPlaying(false);
        });
      return;
    }

    setPlaying(true);
  }

  function toggleVideoPlayback(event) {
    event?.stopPropagation?.();

    if (!canPlayVideo || !videoRef.current) return;

    if (videoRef.current.paused || videoRef.current.ended) {
      startVideoPlayback(event);
      return;
    }

    videoRef.current.pause();
    setPlaying(false);
  }

  function handlePractice(event) {
    event.stopPropagation();
    videoRef.current?.pause();
    setPlaying(false);
    onPractice?.(sign);
  }

  return (
    <article
      className={`sign-card-ui ${variant} ${expanded ? "expanded" : "normal"} ${isPracticeModal ? "practice-modal-card" : ""} ${emphasizeName ? "emphasize-name" : ""} ${largeName ? "large-name-card" : ""} ${className}`}
      style={style}
      draggable={draggable}
      onDragStart={onDragStart}
    >
      {!hideToggle && (
        <button className="card-icon-button" type="button" aria-label={actionLabel} onClick={handleAction}>
          {expanded ? <CloseIcon /> : <FullscreenIcon />}
        </button>
      )}

      {expanded && (
        <div className="screen-top card fade-up">
          <h2>{expandedTitle}</h2>
        </div>
      )}

      <MediaSlot
        sign={sign}
        expanded={expanded}
        videoRef={videoRef}
        playing={playing}
        onTogglePlay={toggleVideoPlayback}
        onVideoEnded={() => {
          setPlaying(false);
          if (videoRef.current) {
            videoRef.current.currentTime = 0;
          }
        }}
        onVideoError={() => {
          setPlaying(false);
          setVideoError(true);
        }}
      />

      {expanded && (
        <button
          className="play-button-ui"
          type="button"
          aria-label="Reproducir video"
          disabled={!canPlayVideo}
          onClick={startVideoPlayback}
        >
          <PlayIcon />
        </button>
      )}

      {expanded && videoError && (
        <small className="practice-save-status error">
          Video no disponible para esta seña.
        </small>
      )}

      {expanded && showPracticeButton && (
        <button
          className="practice-button-ui"
          type="button"
          onClick={handlePractice}
        >
          <span>📷</span>
          Practicar
        </button>
      )}

      {showName && <div className="name-slot">{customNameSlot || sign?.name}</div>}
      {showExpandedName && <div className="name-slot">{sign?.name}</div>}
      {showDescription && (
        <div className="description-slot">
          {sign?.description || "Descripción no disponible para esta seña."}
        </div>
      )}

    </article>
  );
}

function CardPreviewModal({ preview, onClose, onPracticeSign }) {
  const variant = preview.showName || preview.showDescription ? "learning" : "game";
  const canPractice = Boolean(onPracticeSign && preview?.sign?.name);

  return (
    <div className="modal-overlay card-preview-overlay">
      <SignCard
        sign={preview.sign}
        variant={variant}
        expanded
        hideName={!preview.showName}
        showPracticeButton={canPractice}
        onPractice={() => onPracticeSign?.(preview.sign)}
        onCollapse={onClose}
      />
    </div>
  );
}

function CameraPracticeScreen({
  onBack,
  initialLetter = "A",
  singleSignMode = false,
  signs = [],
  onGamificationSync,
  usuarioId = null,
  persistEnabled = true,
}) {
  const letters = useMemo(() => {
    if (singleSignMode && initialLetter) {
      return [initialLetter];
    }

    return signs.map((sign) => sign.name).filter(Boolean);
  }, [initialLetter, signs, singleSignMode]);

  const initialIndex = singleSignMode
    ? 0
    : Math.max(
        0,
        letters.findIndex((letter) => letter === initialLetter)
      );

  const [index, setIndex] = useState(initialIndex);

  useEffect(() => {
    setIndex(initialIndex);
  }, [initialIndex]);

  const current = letters[index] || initialLetter || "A";
  const isLast = index === letters.length - 1;

  function next() {
    if (singleSignMode) return;
    setIndex((prev) => Math.min(prev + 1, letters.length - 1));
  }

  return (
    <div className="page-stack">
      <BackButton onBack={onBack} />

      <div className="screen-header card fade-up">
        <div>
          <h2>Práctica con cámara</h2>
          <p>
            {singleSignMode
              ? `Practicá únicamente la letra ${current} frente a la cámara.`
              : "Realizá la letra indicada frente a la cámara."}
          </p>
        </div>

        <strong>
          {singleSignMode ? `Letra ${current}` : `${index + 1}/${letters.length}`}
        </strong>
      </div>

      {!singleSignMode && (
        <ProgressBar current={index + 1} total={letters.length} />
      )}

      <EvaluatePracticePanel
        targetLabel={current}
        onNext={singleSignMode ? undefined : next}
        isLast={isLast}
        showNextButton={!singleSignMode}
        singleMode={singleSignMode}
        onGamificationSync={onGamificationSync}
        usuarioId={usuarioId}
        persistEnabled={persistEnabled}
      />
    </div>
  );
}

function SpellScreen({ onBack, onGamificationSync, usuarioId = null, persistEnabled = true }) {
  const [spellMode, setSpellMode] = useState("guided");
  const practiceWord = "SOL";

  function handleCompleted(payload) {
    console.log("Palabra completada:", payload);
  }

  return (
    <div className="page-stack">
      <BackButton onBack={onBack} />

      <div className="screen-top card fade-up">
        <div>
          <h2>Deletrear palabras</h2>
          <p className="screen-description">
            Practicá una palabra guiada letra por letra o usá el modo libre para
            formar texto con letras detectadas automáticamente.
          </p>
        </div>
      </div>

      <div className="spell-mode-tabs card fade-up">
        <button
          className={spellMode === "guided" ? "primary" : "secondary"}
          onClick={() => setSpellMode("guided")}
        >
          Deletreo guiado
        </button>

        <button
          className={spellMode === "free" ? "primary" : "secondary"}
          onClick={() => setSpellMode("free")}
        >
          Deletreo libre
        </button>
      </div>

      {spellMode === "guided" && (
        <GuidedSpellPanel
          word={practiceWord}
          onCompleted={handleCompleted}
          onGamificationSync={onGamificationSync}
          usuarioId={usuarioId}
          persistEnabled={persistEnabled}
        />
      )}

      {spellMode === "free" && (
        <FreeSpellPanel 
          onGamificationSync={onGamificationSync}
          persistEnabled={false}
        />
      )}
    </div>
  );
}

function ChallengesScreen({
  category,
  onBack,
  onOpenPreview,
  onGamificationSync,
  onGamificationEvents,
  usuarioId = null,
  persistEnabled = true,
}) {
  const total = 5;
  const [index, setIndex] = useState(0);
  const [feedback, setFeedback] = useState(null);
  const [roundFinished, setRoundFinished] = useState(false);
  const [roundXp, setRoundXp] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);
  const [roundResponses, setRoundResponses] = useState([]);
  const [roundStartedAt, setRoundStartedAt] = useState(() => new Date().toISOString());
  const [roundSaveStatus, setRoundSaveStatus] = useState("idle");
  const [currentAnswer, setCurrentAnswer] = useState(null);

  const challenge = getChallengeForCategory(category.id, index);

  const XP_MINIJUEGO_CORRECTO = 5;
  const XP_BONUS_RONDA_PERFECTA = 10;

  useEffect(() => {
    setCurrentAnswer(null);
    setFeedback(null);
  }, [category.id, index]);

  function handleCheck() {
    if (feedback && feedback !== "missing") return;

    const challengeType = getChallengeForCategory(category.id, index);

    if (!currentAnswer?.isComplete) {
      setFeedback("missing");
      return;
    }

    const isCorrect = Boolean(currentAnswer?.isCorrect);
    const xpForAnswer = isCorrect ? XP_MINIJUEGO_CORRECTO : 0;

    const response = {
      tipo_minijuego: challengeType,
      orden: index + 1,
      fue_correcta: isCorrect,
      xp_obtenida: xpForAnswer,
      respuesta_usuario: currentAnswer?.value ?? null,
    };

    const nextResponses = [...roundResponses, response];
    const nextCorrectCount = correctCount + (isCorrect ? 1 : 0);
    const nextRoundXp = roundXp + xpForAnswer;

    setRoundResponses(nextResponses);
    setFeedback(isCorrect ? "correct" : "wrong");

    if (isCorrect) {
      setRoundXp(nextRoundXp);
      setCorrectCount(nextCorrectCount);
    }

    setTimeout(() => {
      if (index === total - 1) {
        const roundBonus =
          nextCorrectCount === total
            ? XP_BONUS_RONDA_PERFECTA
            : 0;

        const finalRoundXp = nextRoundXp + roundBonus;

        setRoundXp(finalRoundXp);
        setRoundFinished(true);
        persistCompletedRound(nextResponses, nextCorrectCount, finalRoundXp);
      } else {
        setIndex((prev) => prev + 1);
      }

      setFeedback(null);
    }, 1100);
  }

  async function persistCompletedRound(responses, finalCorrectCount, finalXp) {
    setRoundSaveStatus("saving");

    try {
      if (!persistEnabled || !usuarioId) {
        setRoundSaveStatus("saved");
        return;
      }

      const data = await registrarRondaMinijuego({
        usuario_id: usuarioId,
        categoria_id: category.dbId,
        cantidad_minijuegos: total,
        correctas: finalCorrectCount,
      });

      if (Array.isArray(data?.eventos) && data.eventos.length > 0) {
        onGamificationEvents?.(data.eventos);
      }

      await onGamificationSync?.();

      setRoundSaveStatus("saved");
    } catch (error) {
      console.error("No se pudo registrar la ronda de minijuegos:", error);
      setRoundSaveStatus("error");
    }
  }

  if (roundFinished) {
    return (
      <div className="page-stack">
        <BackButton onBack={onBack} />
        <section className="round-summary card fade-up">
          <div className="summary-icon">🎉</div>
          <h2>Ronda completada</h2>
          <p>Terminaste la ronda de desafíos de {category.name}.</p>

          <div className="summary-stats">
            <Stat label="XP ganada" value={`+${roundXp}`} />
            <Stat label="Correctas" value={`${correctCount}/${total}`} />
            <Stat label="Minijuegos" value={total} />
          </div>

          {correctCount === total && (
            <p className="empty-objectives-message">
              Ronda perfecta: +10 XP extra.
            </p>
          )}

          {roundSaveStatus === "saving" && (
            <p className="empty-objectives-message">
              Guardando progreso de la ronda...
            </p>
          )}

          {roundSaveStatus === "error" && (
            <p className="empty-objectives-message">
              No se pudo guardar la ronda.
            </p>
          )}

          <div className="summary-actions">
            <button className="primary" onClick={() => {
              setIndex(0);
              setRoundXp(0);
              setCorrectCount(0);
              setRoundFinished(false);
              setRoundResponses([]);
              setRoundStartedAt(new Date().toISOString());
              setRoundSaveStatus("idle");
              setCurrentAnswer(null);
            }}>
              Jugar otra ronda
            </button>
            <button className="secondary" onClick={onBack}>Volver</button>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <BackButton onBack={onBack} />

      <div className="screen-header card fade-up">
        <div>
          <h2>Desafíos: {category.name}</h2>
          <p>Ronda de minijuegos</p>
        </div>
        <strong>{index + 1}/{total}</strong>
      </div>

      <ProgressBar current={index + 1} total={total} />

      <section key={`${category.id}-${index}`} className="challenge-card challenge-enter">
        {challenge === "signToWord" && <SignToWordChallenge onOpenPreview={onOpenPreview} onAnswerChange={setCurrentAnswer} />}
        {challenge === "wordToSign" && <WordToSignChallenge onOpenPreview={onOpenPreview} onAnswerChange={setCurrentAnswer} />}
        {challenge === "association" && <AssociationChallenge onOpenPreview={onOpenPreview} onAnswerChange={setCurrentAnswer} />}
        {challenge === "complete" && <CompleteChallenge onOpenPreview={onOpenPreview} onAnswerChange={setCurrentAnswer} />}
        {challenge === "phrase" && <PhraseChallenge onOpenPreview={onOpenPreview} onAnswerChange={setCurrentAnswer} />}
        {challenge === "number" && <NumberChallenge onOpenPreview={onOpenPreview} onAnswerChange={setCurrentAnswer} />}
        {challenge === "map" && <MapChallenge onOpenPreview={onOpenPreview} onAnswerChange={setCurrentAnswer} />}

        {feedback && (
          <div className={`challenge-feedback ${feedback}`}>
            {feedback === "correct"
              ? "✔ Correcto  +5 XP"
              : feedback === "missing"
                ? "Completá o seleccioná una respuesta antes de continuar."
                : "✖ Incorrecto"}
          </div>
        )}
      </section>

      <div className="challenge-footer">
        <button className="primary" onClick={handleCheck}>Comprobar y continuar</button>
      </div>
    </div>
  );
}

function getChallengeForCategory(categoryId, index) {
  if (index === 0) return "signToWord";
  if (index === 1) return "wordToSign";
  if (index === 2) return "association";
  if ((categoryId === "comunicacion" || categoryId === "comunicacion_basica") && index === 3) return "phrase";
  if (categoryId === "numeros" && index === 3) return "number";
  if (categoryId === "provincias" && index === 3) return "map";
  return "complete";
}

function SignToWordChallenge({ onOpenPreview, onAnswerChange }) {
  const sign = { id: "hola", name: "Hola", thumb: "👋", description: "Saludo básico." };
  const expected = "Hola";
  const [selectedOption, setSelectedOption] = useState(null);

  function chooseOption(option) {
    setSelectedOption(option);
    onAnswerChange?.({
      isComplete: true,
      isCorrect: option === expected,
      value: option,
    });
  }

  return (
    <div>
      <small className="blue-label">Selección múltiple</small>
      <h3>¿Qué significa esta seña?</h3>

      <div className="challenge-grid">
        <PreviewableSignCard
          sign={sign}
          large
          hideName
          onOpenPreview={onOpenPreview}
          title="Seña a identificar"
        />

        <div className="option-card-grid">
          {["Hola", "Gracias", "Rojo", "Papá"].map((option) => (
            <button
              key={option}
              type="button"
              className={`text-option-card ${selectedOption === option ? "selected" : ""}`}
              onClick={() => chooseOption(option)}
            >
              {option}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function WordToSignChallenge({ onOpenPreview, onAnswerChange }) {
  const [selectedId, setSelectedId] = useState(null);
  const expectedId = "hola";
  const options = [
    { id: "hola", name: "Hola", thumb: "👋", description: "Saludo básico." },
    { id: "gracias", name: "Gracias", thumb: "🙏", description: "Expresión de agradecimiento." },
    { id: "mama", name: "Mamá", thumb: "👩", description: "Seña correspondiente a mamá." },
    { id: "azul", name: "Azul", thumb: "🔵", description: "Seña correspondiente al color azul." },
  ];

  function chooseSign(sign) {
    setSelectedId(sign.id);
    onAnswerChange?.({
      isComplete: true,
      isCorrect: sign.id === expectedId,
      value: sign.name,
    });
  }

  return (
    <div>
      <small className="blue-label">Selección múltiple</small>
      <h3>Elegí la seña correspondiente a: <span>Hola</span></h3>

      <div className="sign-option-grid">
        {options.map((sign) => (
          <div
            key={sign.id}
            className={`choice-card ${selectedId === sign.id ? "selected" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => chooseSign(sign)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") chooseSign(sign);
            }}
          >
            <PreviewableSignCard sign={sign} hideName onOpenPreview={onOpenPreview} />
          </div>
        ))}
      </div>
    </div>
  );
}

function AssociationChallenge({ onOpenPreview, onAnswerChange }) {
  const signs = [
    { id: "rojo", name: "Rojo", thumb: "🔴", description: "Seña correspondiente al color rojo." },
    { id: "azul", name: "Azul", thumb: "🔵", description: "Seña correspondiente al color azul." },
    { id: "verde", name: "Verde", thumb: "🟢", description: "Seña correspondiente al color verde." },
  ];

  const [answers, setAnswers] = useState({});

  function handleDropName(signId, label, fromSlot) {
    setAnswers((prev) => {
      const next = { ...prev };
      if (fromSlot && fromSlot !== signId) next[fromSlot] = null;
      Object.keys(next).forEach((slot) => {
        if (slot !== signId && next[slot] === label) next[slot] = null;
      });
      next[signId] = label;
      return next;
    });
  }

  function clearNameSlot(slotId) {
    setAnswers((prev) => ({ ...prev, [slotId]: null }));
  }

  const usedNames = Object.values(answers).filter(Boolean);
  const availableNames = signs.filter((sign) => !usedNames.includes(sign.name));

  useEffect(() => {
    const isComplete = signs.every((sign) => Boolean(answers[sign.id]));
    const isCorrect = isComplete && signs.every((sign) => answers[sign.id] === sign.name);
    onAnswerChange?.({
      isComplete,
      isCorrect,
      value: { ...answers },
    });
  }, [answers, onAnswerChange]);

  return (
    <div>
      <small className="blue-label">Asociación</small>
      <h3>Arrastrá cada nombre a su tarjeta de seña</h3>

      <div className="association-grid">
        {signs.map((sign) => (
          <div key={sign.id} className="association-card clean">
            <SignCard
              sign={sign}
              variant="association"
              expanded={false}
              hideName
              customNameSlot={
                <NameDropArea
                  slotId={sign.id}
                  value={answers[sign.id]}
                  onDropName={(label, fromSlot) => handleDropName(sign.id, label, fromSlot)}
                />
              }
              onExpand={() => onOpenPreview({ sign, title: "Vista de seña", showName: false, showDescription: false })}
            />
          </div>
        ))}
      </div>

      <SignOptionsTray onReturn={clearNameSlot}>
        {availableNames.map((sign) => (
          <DragName key={sign.id} label={sign.name} />
        ))}
      </SignOptionsTray>
    </div>
  );
}

function CompleteChallenge({ onOpenPreview, onAnswerChange }) {
  const [answer, setAnswer] = useState(null);

  function placeAnswer(item, fromSlot) {
    setAnswer(item);
  }

  const options = [
    { id: "futbol", name: "Fútbol", thumb: "⚽", description: "Seña correspondiente a fútbol." },
    { id: "basquet", name: "Básquet", thumb: "🏀", description: "Seña correspondiente a básquet." },
    { id: "tenis", name: "Tenis", thumb: "🎾", description: "Seña correspondiente a tenis." },
  ];

  const availableOptions = options.filter((sign) => sign.id !== answer?.id);

  useEffect(() => {
    onAnswerChange?.({
      isComplete: Boolean(answer),
      isCorrect: answer?.id === "futbol",
      value: answer?.name ?? null,
    });
  }, [answer, onAnswerChange]);

  return (
    <div>
      <small className="blue-label">Completar con seña</small>
      <h3>Completá la consigna según el contexto</h3>

      <div className="context-box">
        <div className="context-image">⚽</div>
        <p>Me gusta jugar al</p>
        <DropZone
          label={answer ? "" : "Soltar seña"}
          slotId="complete-answer"
          onDropItem={placeAnswer}
          card={answer}
          onOpenPreview={onOpenPreview}
        />
        <p>los fines de semana.</p>
      </div>

      <SignOptionsTray onReturn={(slotId) => slotId === "complete-answer" && setAnswer(null)}>
        {availableOptions.map((sign) => (
          <DragSignCard key={sign.id} sign={sign} onOpenPreview={onOpenPreview} />
        ))}
      </SignOptionsTray>
    </div>
  );
}

function PhraseChallenge({ onOpenPreview, onAnswerChange }) {
  const [slots, setSlots] = useState({ slot1: null, slot2: null, slot3: null });

  const options = [
    { id: "hola", name: "Hola", thumb: "👋", description: "Saludo básico." },
    { id: "me-llamo", name: "Me llamo", thumb: "🙋", description: "Expresión para presentarse." },
    { id: "como-estas", name: "¿Cómo estás?", thumb: "❓", description: "Pregunta básica." },
  ];

  const usedIds = Object.values(slots).filter(Boolean).map((sign) => sign.id);
  const availableOptions = options.filter((sign) => !usedIds.includes(sign.id));

  function setSlot(slotId, item, fromSlot) {
    setSlots((prev) => {
      const next = { ...prev };
      if (fromSlot && fromSlot !== slotId) next[fromSlot] = null;
      Object.keys(next).forEach((slot) => {
        if (slot !== slotId && next[slot]?.id === item?.id) next[slot] = null;
      });
      next[slotId] = item;
      return next;
    });
  }

  function clearSlot(slotId) {
    setSlots((prev) => ({ ...prev, [slotId]: null }));
  }

  useEffect(() => {
    const expected = { slot1: "hola", slot2: "me-llamo", slot3: "como-estas" };
    const isComplete = Object.keys(expected).every((slotId) => Boolean(slots[slotId]));
    const isCorrect = isComplete && Object.entries(expected).every(([slotId, expectedId]) => slots[slotId]?.id === expectedId);
    onAnswerChange?.({
      isComplete,
      isCorrect,
      value: Object.fromEntries(Object.entries(slots).map(([slotId, sign]) => [slotId, sign?.name ?? null])),
    });
  }, [slots, onAnswerChange]);

  return (
    <div>
      <small className="blue-label">Armar frase básica</small>
      <h3>Hola, me llamo Pedro, ¿cómo estás?</h3>

      <div className="phrase-row">
        <DropZone label="Seña" slotId="slot1" onDropItem={(item, fromSlot) => setSlot("slot1", item, fromSlot)} card={slots.slot1} onOpenPreview={onOpenPreview} />
        <DropZone label="Seña" slotId="slot2" onDropItem={(item, fromSlot) => setSlot("slot2", item, fromSlot)} card={slots.slot2} onOpenPreview={onOpenPreview} />
        <span className="fixed-word">Pedro</span>
        <DropZone label="Seña" slotId="slot3" onDropItem={(item, fromSlot) => setSlot("slot3", item, fromSlot)} card={slots.slot3} onOpenPreview={onOpenPreview} />
      </div>

      <SignOptionsTray onReturn={clearSlot}>
        {availableOptions.map((sign) => (
          <DragSignCard key={sign.id} sign={sign} onOpenPreview={onOpenPreview} />
        ))}
      </SignOptionsTray>
    </div>
  );
}

function NumberChallenge({ onOpenPreview, onAnswerChange }) {
  const [slots, setSlots] = useState({ slot1: null, slot2: null });

  const options = [
    { id: "3", name: "Tres", thumb: "3", description: "Seña correspondiente al número 3." },
    { id: "4", name: "Cuatro", thumb: "4", description: "Seña correspondiente al número 4." },
  ];

  const usedIds = Object.values(slots).filter(Boolean).map((sign) => sign.id);
  const availableOptions = options.filter((sign) => !usedIds.includes(sign.id));

  function setSlot(slotId, item, fromSlot) {
    setSlots((prev) => {
      const next = { ...prev };
      if (fromSlot && fromSlot !== slotId) next[fromSlot] = null;
      Object.keys(next).forEach((slot) => {
        if (slot !== slotId && next[slot]?.id === item?.id) next[slot] = null;
      });
      next[slotId] = item;
      return next;
    });
  }

  function clearSlot(slotId) {
    setSlots((prev) => ({ ...prev, [slotId]: null }));
  }

  useEffect(() => {
    const expected = { slot1: "3", slot2: "4" };
    const isComplete = Object.keys(expected).every((slotId) => Boolean(slots[slotId]));
    const isCorrect = isComplete && Object.entries(expected).every(([slotId, expectedId]) => slots[slotId]?.id === expectedId);
    onAnswerChange?.({
      isComplete,
      isCorrect,
      value: Object.fromEntries(Object.entries(slots).map(([slotId, sign]) => [slotId, sign?.name ?? null])),
    });
  }, [slots, onAnswerChange]);

  return (
    <div>
      <small className="blue-label">Ordenar números</small>
      <h3>Completá la secuencia</h3>

      <div className="phrase-row">
        <span className="fixed-word">2</span>
        <DropZone label="Seña" slotId="slot1" onDropItem={(item, fromSlot) => setSlot("slot1", item, fromSlot)} card={slots.slot1} onOpenPreview={onOpenPreview} />
        <DropZone label="Seña" slotId="slot2" onDropItem={(item, fromSlot) => setSlot("slot2", item, fromSlot)} card={slots.slot2} onOpenPreview={onOpenPreview} />
        <span className="fixed-word">5</span>
      </div>

      <SignOptionsTray onReturn={clearSlot}>
        {availableOptions.map((sign) => (
          <DragSignCard key={sign.id} sign={sign} onOpenPreview={onOpenPreview} />
        ))}
      </SignOptionsTray>
    </div>
  );
}

function MapChallenge({ onOpenPreview, onAnswerChange }) {
  const mapTargets = useMemo(() => [
    {
      id: "salta",
      provinceId: "ARA",
      name: "Salta",
      thumb: "S",
      description: "Seña correspondiente a la provincia de Salta.",
    },
    {
      id: "santiago_del_estero",
      provinceId: "ARG",
      name: "Santiago del Estero",
      thumb: "SE",
      description: "Seña correspondiente a la provincia de Santiago del Estero.",
    },
    {
      id: "la_rioja",
      provinceId: "ARF",
      name: "La Rioja",
      thumb: "LR",
      description: "Seña correspondiente a la provincia de La Rioja.",
    },
  ], []);

  const [slots, setSlots] = useState(() =>
    Object.fromEntries(mapTargets.map((target) => [target.id, null]))
  );

  const usedIds = Object.values(slots).filter(Boolean).map((sign) => sign.id);
  const availableOptions = mapTargets.filter((sign) => !usedIds.includes(sign.id));

  function setSlot(slotId, item, fromSlot) {
    setSlots((prev) => {
      const next = { ...prev };
      if (fromSlot && fromSlot !== slotId) next[fromSlot] = null;
      Object.keys(next).forEach((slot) => {
        if (slot !== slotId && next[slot]?.id === item?.id) next[slot] = null;
      });
      next[slotId] = item;
      return next;
    });
  }

  function clearSlot(slotId) {
    setSlots((prev) => ({ ...prev, [slotId]: null }));
  }

  useEffect(() => {
    const isComplete = mapTargets.every((target) => Boolean(slots[target.id]));
    const isCorrect = isComplete && mapTargets.every((target) => slots[target.id]?.id === target.id);
    onAnswerChange?.({
      isComplete,
      isCorrect,
      value: Object.fromEntries(Object.entries(slots).map(([slotId, sign]) => [slotId, sign?.name ?? null])),
    });
  }, [slots, mapTargets, onAnswerChange]);

  return (
    <div>
      <small className="blue-label">Mapa de provincias</small>
      <h3>Ubicá las señas en el mapa</h3>

      <div className="map-layout precise-map-layout">
        <InteractiveArgentinaMap
          targets={mapTargets}
          slots={slots}
          onDropTarget={setSlot}
          onOpenPreview={onOpenPreview}
        />

        <SignOptionsTray onReturn={clearSlot}>
          {availableOptions.map((sign) => (
            <DragSignCard key={sign.id} sign={sign} onOpenPreview={onOpenPreview} />
          ))}
        </SignOptionsTray>
      </div>
    </div>
  );
}

function InteractiveArgentinaMap({ targets, slots, onDropTarget, onOpenPreview }) {
  const hostRef = useRef(null);
  const [mapSize, setMapSize] = useState({ width: 1000, height: 620 });

  const provincePoints = useMemo(() => ({
    ARA: { x: 463.3, y: 124.5 }, // Salta
    ARG: { x: 506.7, y: 180.8 }, // Santiago del Estero
    ARF: { x: 429.4, y: 223.5 }, // La Rioja
    ARY: { x: 452.1, y: 68.9 },
    ART: { x: 465.2, y: 160.5 },
    ARK: { x: 423.8, y: 168.7 },
    ARX: { x: 500.2, y: 279.7 },
    ARJ: { x: 391.6, y: 254.2 },
    ARM: { x: 401.6, y: 345.1 },
  }), []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || typeof ResizeObserver === "undefined") return;

    const updateSize = () => {
      const rect = host.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setMapSize((prev) => {
          const width = Math.round(rect.width);
          const height = Math.round(rect.height);
          if (Math.abs(prev.width - width) < 2 && Math.abs(prev.height - height) < 2) return prev;
          return { width, height };
        });
      }
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const mapModel = useMemo(() => {
    const points = targets
      .map((target) => ({ target, point: provincePoints[target.provinceId] }))
      .filter((entry) => entry.point);

    if (!points.length) {
      return {
        viewBox: { x: 330, y: 40, width: 360, height: 260 },
        positions: Object.fromEntries(targets.map((target, index) => [
          target.id,
          { left: 20 + index * 16, top: 20 + index * 12, w: 46, h: 62 },
        ])),
      };
    }

    const xs = points.map((entry) => entry.point.x);
    const ys = points.map((entry) => entry.point.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    const aspect = mapSize.width / Math.max(mapSize.height, 1);
    const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1.65;

    // El objetivo es mostrar la región donde están las provincias, no Argentina completa.
    // Partimos del grupo de puntos de etiqueta que ya trae SimpleMaps y agregamos un margen
    // moderado para que se entienda la ubicación relativa sin perder el enfoque.
    const groupWidth = Math.max(maxX - minX, 1);
    const groupHeight = Math.max(maxY - minY, 1);
    let width = Math.max(groupWidth + 190, 360);
    let height = Math.max(groupHeight + 150, 260);

    if (width / height < safeAspect) width = height * safeAspect;
    if (width / height > safeAspect) height = width / safeAspect;

    width = Math.min(width, 1000);
    height = Math.min(height, 1000);

    const x = Math.max(0, Math.min(centerX - width / 2, 1000 - width));
    const y = Math.max(0, Math.min(centerY - height / 2, 1000 - height));
    const viewBox = { x, y, width, height };

    const positions = Object.fromEntries(
      targets.map((target) => {
        const point = provincePoints[target.provinceId];
        if (!point) return [target.id, { left: 50, top: 50, w: 46, h: 62 }];

        return [target.id, {
          left: ((point.x - viewBox.x) / viewBox.width) * 100,
          top: ((point.y - viewBox.y) / viewBox.height) * 100,
          w: 48,
          h: 64,
        }];
      })
    );

    return { viewBox, positions };
  }, [targets, provincePoints, mapSize.width, mapSize.height]);

  const vb = mapModel.viewBox;

  return (
    <div className="map-box precise-argentina-map">
      <div className="map-note">Mapa interactivo de provincias</div>

      <div className="interactive-argentina-svg-host" ref={hostRef}>
        <svg
          className="argentina-map-svg-image"
          viewBox={`${vb.x} ${vb.y} ${vb.width} ${vb.height}`}
          preserveAspectRatio="xMidYMid meet"
          aria-label="Mapa de provincias de Argentina"
        >
          <image
            href="/assets/maps/argentina.svg"
            x="0"
            y="0"
            width="1000"
            height="1000"
            preserveAspectRatio="xMidYMid meet"
          />
        </svg>

        {targets.map((target) => {
          const position = mapModel.positions[target.id] || { left: 50, top: 50, w: 48, h: 64 };
          return (
            <div
              key={target.id}
              className="province-drop-anchor"
              style={{
                left: `${position.left}%`,
                top: `${position.top}%`,
                width: `${position.w}px`,
                height: `${position.h}px`,
              }}
            >
              <DropZone
                compact
                label={target.name}
                slotId={target.id}
                onDropItem={(item, fromSlot) => onDropTarget(target.id, item, fromSlot)}
                card={slots[target.id]}
                onOpenPreview={onOpenPreview}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PreviewableSignCard({ sign, onOpenPreview, hideName = false, title }) {
  return (
    <div className="preview-card-center">
      <SignCard
        sign={sign}
        variant={hideName ? "game" : "learning"}
        expanded={false}
        hideName={hideName}
        onExpand={() =>
          onOpenPreview({
            sign,
            title: title || "Vista de seña",
            showName: !hideName,
            showDescription: false,
          })
        }
      />
    </div>
  );
}

function DragSignCard({ sign, onOpenPreview }) {
  function handleDragStart(event) {
    event.dataTransfer.setData("application/json", JSON.stringify({ type: "sign", sign }));
  }

  return (
    <div className="drag-sign-card-wrap">
      <SignCard
        sign={sign}
        variant="game"
        expanded={false}
        className="drag-sign-card"
        draggable
        onDragStart={handleDragStart}
        onExpand={() => onOpenPreview({ sign, title: "Vista de seña", showName: false, showDescription: false })}
      />
    </div>
  );
}

function DragName({ label }) {
  function handleDragStart(event) {
    event.dataTransfer.setData("application/json", JSON.stringify({ type: "name", label }));
  }

  return (
    <div className="drag-name" draggable onDragStart={handleDragStart}>
      {label}
    </div>
  );
}

function SignOptionsTray({ children, onReturn }) {
  function handleDragOver(event) {
    event.preventDefault();
  }

  function handleDrop(event) {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/json");
    if (!raw) return;
    const payload = JSON.parse(raw);
    if (payload?.fromSlot) onReturn?.(payload.fromSlot);
  }

  return (
    <div className="drag-row sign-options-tray" onDragOver={handleDragOver} onDrop={handleDrop}>
      {children}
    </div>
  );
}

function DropZone({ label, onDropItem, card = null, compact = false, slotId, onOpenPreview }) {
  function handleDragOver(event) {
    event.preventDefault();
  }

  function handleDrop(event) {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/json");
    if (!raw) return;
    const payload = JSON.parse(raw);
    if (payload?.type !== "sign" || !payload.sign) return;
    onDropItem(payload.sign, payload.fromSlot || null);
  }

  function handleCardDragStart(event) {
    event.dataTransfer.setData("application/json", JSON.stringify({ type: "sign", sign: card, fromSlot: slotId }));
  }

  return (
    <div
      className={compact ? "drop-zone compact" : "drop-zone"}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {card ? (
        <SignCard
          sign={card}
          variant="game"
          expanded={false}
          draggable
          onDragStart={handleCardDragStart}
          onExpand={() => onOpenPreview?.({ sign: card, title: "Vista de seña", showName: false, showDescription: false })}
          className={compact ? "drop-zone-card compact" : "drop-zone-card"}
        />
      ) : (
        <span className="drop-zone-label">{label}</span>
      )}
    </div>
  );
}

function NameDropArea({ value, onDropName, slotId }) {
  function handleDragOver(event) {
    event.preventDefault();
  }

  function handleDrop(event) {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/json");
    if (!raw) return;
    const payload = JSON.parse(raw);
    if (payload?.type === "name") onDropName(payload.label, payload.fromSlot || null);
  }

  function handleDragStart(event) {
    if (!value) return;
    event.dataTransfer.setData("application/json", JSON.stringify({ type: "name", label: value, fromSlot: slotId }));
  }

  return (
    <div
      className={`name-drop-area ${value ? "filled" : ""}`}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      draggable={Boolean(value)}
      onDragStart={handleDragStart}
      title={value ? "Arrastrar para mover o devolver a opciones" : undefined}
    >
      {value || "Soltar nombre"}
    </div>
  );
}

function ProfileScreen({ isGuest, profileData, setProfileData, onUpdateProfile, usuarioId, userProgress, onBack, onStats, onAchievements }) {
  const [editingField, setEditingField] = useState(null);
  const [draftValue, setDraftValue] = useState("");
  const [draftPassword, setDraftPassword] = useState("");
  const [draftPasswordConfirm, setDraftPasswordConfirm] = useState("");
  const [profileError, setProfileError] = useState("");
  const [profileSaving, setProfileSaving] = useState(false);
  const [customizerOpen, setCustomizerOpen] = useState(false);
  const [customizerTab, setCustomizerTab] = useState("frames");

  const [recentAchievements, setRecentAchievements] = useState([]);
  const [recentAchievementsStatus, setRecentAchievementsStatus] = useState("loading");
  const [profileStats, setProfileStats] = useState({
    seniasAprendidas: 0,
    rondasCompletadas: 0,
    palabrasDeletreadas: 0,
  });

  useEffect(() => {
    if (isGuest || !usuarioId) {
      setProfileStats({ seniasAprendidas: 0, rondasCompletadas: 0, palabrasDeletreadas: 0 });
      return;
    }

    let cancelled = false;

    async function loadProfileStats() {
      try {
        const panel = await obtenerPanelUsuario(usuarioId);
        if (cancelled) return;

        const rondasPorCategoria = panel?.rondas_por_categoria || {};
        const totalRondas = Object.values(rondasPorCategoria).reduce(
          (sum, value) => sum + Number(value || 0),
          0
        );

        setProfileStats({
          seniasAprendidas: Number(panel?.senias_aprendidas_camara || 0),
          rondasCompletadas: totalRondas,
          palabrasDeletreadas: Number(panel?.palabras_deletreadas_exitosamente || 0),
        });
      } catch (error) {
        console.warn("No se pudieron cargar las estadísticas del perfil:", error);
      }
    }

    loadProfileStats();

    return () => {
      cancelled = true;
    };
  }, [isGuest, usuarioId]);

  useEffect(() => {
    if (isGuest) return;

    let cancelled = false;

    async function loadRecentAchievements() {
      setRecentAchievementsStatus("loading");

      try {
        if (isGuest || !usuarioId) {
          if (cancelled) return;
          setRecentAchievements([]);
          setRecentAchievementsStatus("ready");
          return;
        }

        const data = await obtenerLogrosUsuario(usuarioId);

        if (cancelled) return;

        const unlockedAchievements = normalizeAchievements(data.logros)
          .filter((achievement) => achievement.unlocked)
          .sort((a, b) => {
            const dateA = a.unlockedAt ? new Date(a.unlockedAt).getTime() : 0;
            const dateB = b.unlockedAt ? new Date(b.unlockedAt).getTime() : 0;
            return dateB - dateA;
          })
          .slice(0, 3);

        setRecentAchievements(unlockedAchievements);
        setRecentAchievementsStatus("ready");
      } catch (error) {
        console.error(error);

        if (!cancelled) {
          setRecentAchievements([]);
          setRecentAchievementsStatus("error");
        }
      }
    }

    loadRecentAchievements();

    return () => {
      cancelled = true;
    };
  }, [isGuest, usuarioId]);

  const selectedFrame = frameFromProfileData(profileData);

  function startEditing(field, value = "") {
    setProfileError("");
    setEditingField(field);
    setDraftValue(value);
    setDraftPassword("");
    setDraftPasswordConfirm("");
  }

  async function saveField() {
    if (!editingField) return;

    const normalizedValue = String(draftValue || "").trim();

    if (editingField === "name" && !normalizedValue) {
      setProfileError("El nombre no puede quedar vacío.");
      return;
    }

    if (editingField === "email") {
      const emailPattern = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
      if (!normalizedValue) {
        setProfileError("El correo electrónico no puede quedar vacío.");
        return;
      }
      if (!emailPattern.test(normalizedValue)) {
        setProfileError("El correo electrónico no tiene un formato válido.");
        return;
      }
    }

    if (editingField === "password") {
      if (!draftPassword) {
        setProfileError("Ingresá la nueva contraseña.");
        return;
      }
      if (draftPassword.length < 8) {
        setProfileError("La contraseña debe tener al menos 8 caracteres.");
        return;
      }
      if (!draftPasswordConfirm) {
        setProfileError("Confirmá la nueva contraseña.");
        return;
      }
      if (draftPassword !== draftPasswordConfirm) {
        setProfileError("Las contraseñas no coinciden.");
        return;
      }
    }

    setProfileSaving(true);
    setProfileError("");

    try {
      if (editingField === "name") {
        await onUpdateProfile?.({ nombre_visible: normalizedValue });
      }

      if (editingField === "email") {
        await onUpdateProfile?.({ email: normalizedValue.toLowerCase() });
      }

      if (editingField === "password") {
        await onUpdateProfile?.({ password: draftPassword });
      }

      setEditingField(null);
      setDraftValue("");
      setDraftPassword("");
      setDraftPasswordConfirm("");
    } catch (error) {
      setProfileError(error.message || "No se pudieron guardar los cambios del perfil.");
    } finally {
      setProfileSaving(false);
    }
  }

  function cancelEditing() {
    setEditingField(null);
    setDraftValue("");
    setDraftPassword("");
    setDraftPasswordConfirm("");
  }

  function openCustomizer(tab) {
    setCustomizerTab(tab);
    setCustomizerOpen(true);
  }

  if (isGuest) {
    return (
      <div className="page-stack">
        <BackButton onBack={onBack} />
        <section className="empty-card fade-up">
          <h2>Modo invitado</h2>
          <p>Iniciá sesión para modificar tu perfil y guardar tu progreso.</p>
        </section>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <BackButton onBack={onBack} />

      <div className="screen-top card fade-up">
        <div>
          <h2>Perfil</h2>
          <p className="screen-description">
            Consultá tus datos personales, estadísticas recientes y últimos logros.
          </p>
        </div>
      </div>

      <div className="profile-modern-layout">
        <section className="profile-identity-card card fade-up">
          <div className="profile-identity-preview">
            <IdentityFrame
              photo={profileData.photo}
              frame={selectedFrame}
              size="large"
              editable
              onEdit={() => openCustomizer("photos")}
            />

            <div className="identity-title-block">
              <h3>{profileData.name}</h3>
              <p>{profileData.title}</p>
            </div>
          </div>

          <div className="profile-level-mini">
            <span>{`Nivel ${userProgress?.nivel ?? 1}`}</span>
            <strong>{userProgress?.xpNivelActual ?? 0}/{userProgress?.xpSiguienteNivel ?? 120} XP</strong>
            <div className="progress slim">
              <div style={{ width: `${Math.min(((userProgress?.xpNivelActual ?? 0) / Math.max(userProgress?.xpSiguienteNivel ?? 120, 1)) * 100, 100)}%` }} />
            </div>
          </div>

        </section>

        <section className="profile-details-card card fade-up delay-1">
          <div className="section-heading compact">
            <div>
              <h3>Datos del usuario</h3>
              <p>Actualizá tu nombre, correo electrónico, contraseña y foto de perfil.</p>
            </div>
          </div>

          <div className="profile-detail-list">
            {profileError && <p className="auth-error">{profileError}</p>}

            <EditableFieldRow
              label="Nombre"
              value={profileData.name}
              isEditing={editingField === "name"}
              draftValue={draftValue}
              onEdit={() => startEditing("name", profileData.name)}
              onChange={setDraftValue}
              onSave={saveField}
              onCancel={cancelEditing}
              saving={profileSaving}
            />

            <EditableFieldRow
              label="Correo electrónico"
              value={profileData.email}
              isEditing={editingField === "email"}
              draftValue={draftValue}
              onEdit={() => startEditing("email", profileData.email)}
              onChange={setDraftValue}
              onSave={saveField}
              onCancel={cancelEditing}
              saving={profileSaving}
              inputType="email"
            />

            <EditablePasswordRow
              isEditing={editingField === "password"}
              passwordValue={draftPassword}
              confirmValue={draftPasswordConfirm}
              onEdit={() => startEditing("password")}
              onPasswordChange={setDraftPassword}
              onConfirmChange={setDraftPasswordConfirm}
              onSave={saveField}
              onCancel={cancelEditing}
              saving={profileSaving}
            />
          </div>
        </section>
      </div>

      <div className="profile-extra-grid">
        <section className="profile-extra-card card fade-up delay-1">
          <div className="home-panel-header">
            <div>
              <h3>Últimas estadísticas</h3>
              <p>Resumen rápido de tu actividad.</p>
            </div>
            <button className="text-link-button" onClick={onStats}>Ver todas</button>
          </div>

          <div className="home-stat-row">
            <Stat label="Señas aprendidas" value={profileStats.seniasAprendidas} />
            <Stat label="Rondas completadas" value={profileStats.rondasCompletadas} />
            <Stat label="Palabras deletreadas" value={profileStats.palabrasDeletreadas} />
          </div>
        </section>

        <section className="profile-extra-card card fade-up delay-2">
          <div className="home-panel-header">
            <div>
              <h3>Últimos logros</h3>
              <p>Medallas desbloqueadas recientemente.</p>
            </div>
            <button className="text-link-button" onClick={onAchievements}>Ver todos</button>
          </div>

          <div className="home-achievement-row">
            {recentAchievementsStatus === "loading" && (
              <p className="empty-objectives-message">
                Cargando logros...
              </p>
            )}

            {recentAchievementsStatus === "error" && (
              <p className="empty-objectives-message">
                No se pudieron cargar los últimos logros.
              </p>
            )}

            {recentAchievementsStatus === "ready" && recentAchievements.length === 0 && (
              <p className="empty-objectives-message">
                Aún no desbloqueaste logros.
              </p>
            )}

            {recentAchievementsStatus === "ready" &&
              recentAchievements.map((achievement) => (
                <Badge
                  key={achievement.codigo || achievement.id}
                  achievement={achievement}
                />
              ))}
          </div>

        </section>
      </div>

      {customizerOpen && (
        <IdentityCustomizationModal
          profileData={profileData}
          setProfileData={setProfileData}
          selectedFrame={selectedFrame}
          usuarioId={usuarioId}
          onUpdateProfile={onUpdateProfile}
          activeTab={customizerTab}
          setActiveTab={setCustomizerTab}
          onClose={() => setCustomizerOpen(false)}
        />
      )}
    </div>
  );
}

function EditableFieldRow({
  label,
  value,
  isEditing,
  draftValue,
  onEdit,
  onChange,
  onSave,
  onCancel,
  readOnly = false,
  saving = false,
  inputType = "text",
}) {
  return (
    <div className="profile-detail-row">
      <div className="profile-detail-info">
        <span>{label}</span>

        {!isEditing ? (
          <strong>{value}</strong>
        ) : (
          <div className="profile-inline-edit">
            <input
              type={inputType}
              value={draftValue}
              onChange={(e) => onChange(e.target.value)}
            />
            <div className="profile-inline-actions">
              <button className="mini-save" onClick={onSave} disabled={saving}>
                {saving ? "Guardando..." : "Guardar"}
              </button>
              <button className="mini-cancel" onClick={onCancel} disabled={saving}>Cancelar</button>
            </div>
          </div>
        )}
      </div>

      {!isEditing && !readOnly && (
        <button className="detail-edit-button" onClick={onEdit} title={`Editar ${label}`}>
          ✎
        </button>
      )}
    </div>
  );
}

function EditablePasswordRow({
  isEditing,
  passwordValue,
  confirmValue,
  onEdit,
  onPasswordChange,
  onConfirmChange,
  onSave,
  onCancel,
  saving = false,
}) {
  return (
    <div className="profile-detail-row">
      <div className="profile-detail-info">
        <span>Contraseña</span>

        {!isEditing ? (
          <strong>••••••••</strong>
        ) : (
          <div className="profile-inline-edit profile-password-edit">
            <input
              type="password"
              value={passwordValue}
              onChange={(event) => onPasswordChange(event.target.value)}
              placeholder="Nueva contraseña"
              autoComplete="new-password"
            />
            <input
              type="password"
              value={confirmValue}
              onChange={(event) => onConfirmChange(event.target.value)}
              placeholder="Confirmar contraseña"
              autoComplete="new-password"
            />
            <div className="profile-inline-actions">
              <button className="mini-save" onClick={onSave} disabled={saving}>
                {saving ? "Guardando..." : "Guardar"}
              </button>
              <button className="mini-cancel" onClick={onCancel} disabled={saving}>
                Cancelar
              </button>
            </div>
          </div>
        )}
      </div>

      {!isEditing && (
        <button className="detail-edit-button" onClick={onEdit} title="Editar contraseña">
          ✎
        </button>
      )}
    </div>
  );
}

function IdentityCustomizationModal({
  profileData,
  setProfileData,
  selectedFrame,
  activeTab,
  setActiveTab,
  usuarioId,
  onUpdateProfile,
  onClose,
}) {
  const [sourcePhoto, setSourcePhoto] = useState(null);
  const [crop, setCrop] = useState({ x: 0, y: 0, zoom: 1 });
  const [isDraggingCrop, setIsDraggingCrop] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const fileInputRef = useRef(null);
  const [frameOptions, setFrameOptions] = useState([]);
  const [titleOptions, setTitleOptions] = useState([]);

  useEffect(() => {
    let cancelled = false;

    async function loadCustomizationOptions() {
      try {
        const [marcos, titulos] = await Promise.all([
          obtenerMarcos(usuarioId),
          obtenerTitulos(usuarioId),
        ]);

        if (cancelled) return;

        const normalizedFrames = Array.isArray(marcos)
          ? marcos.map(normalizeFrameOptionFromApi).sort((a, b) => a.order - b.order)
          : [];
        const normalizedTitles = Array.isArray(titulos)
          ? titulos.map(normalizeTitleOptionFromApi).sort((a, b) => a.order - b.order)
          : [];

        setFrameOptions(normalizedFrames);
        setTitleOptions(normalizedTitles);

        setProfileData((prev) => {
          const selectedFrameStillExists = normalizedFrames.some((frame) => frame.id === prev.frame || frame.dbId === prev.frameDbId);
          const selectedTitleStillExists = normalizedTitles.some((title) => title.name === prev.title || title.dbId === prev.titleDbId);
          const firstAvailableFrame = normalizedFrames.find((frame) => frame.disponible) || normalizedFrames[0];
          const firstAvailableTitle = normalizedTitles.find((title) => title.disponible) || normalizedTitles[0];

          return {
            ...prev,
            frame: selectedFrameStillExists ? prev.frame : (firstAvailableFrame?.id || prev.frame),
            frameDbId: selectedFrameStillExists ? prev.frameDbId : (firstAvailableFrame?.dbId || prev.frameDbId),
            frameName: selectedFrameStillExists ? prev.frameName : (firstAvailableFrame?.name || prev.frameName),
            frameImageUrl: selectedFrameStillExists ? prev.frameImageUrl : (firstAvailableFrame?.imageUrl || prev.frameImageUrl),
            title: selectedTitleStillExists ? prev.title : (firstAvailableTitle?.name || prev.title),
            titleDbId: selectedTitleStillExists ? prev.titleDbId : (firstAvailableTitle?.dbId || prev.titleDbId),
          };
        });
      } catch (error) {
        console.warn("No se pudieron cargar marcos o títulos desde la base de datos:", error);
      }
    }

    loadCustomizationOptions();

    return () => {
      cancelled = true;
    };
  }, [usuarioId, setProfileData]);

  const availableFrameOptions = frameOptions;
  const availableTitleOptions = titleOptions;

  async function persistEquipment(nextFrame, nextTitle) {
    if (!usuarioId || !nextFrame?.dbId || !nextTitle?.dbId) return;

    try {
      await equiparPerfil({
        usuario_id: usuarioId,
        marco_id: nextFrame.dbId,
        titulo_id: nextTitle.dbId,
      });
    } catch (error) {
      console.warn("No se pudo guardar la personalización del perfil:", error);
    }
  }

  async function persistPhoto(nextPhoto) {
    setProfileData((prev) => ({
      ...prev,
      photo: nextPhoto,
    }));

    try {
      await onUpdateProfile?.({ foto_perfil_url: nextPhoto });
    } catch (error) {
      console.warn("No se pudo guardar la foto de perfil:", error);
    }
  }


  function selectFrame(frame) {
    const currentTitle = availableTitleOptions.find((title) => title.dbId === profileData.titleDbId || title.name === profileData.title) || availableTitleOptions[0];

    setProfileData((prev) => ({
      ...prev,
      frame: frame.id,
      frameDbId: frame.dbId,
      frameName: frame.name,
      frameImageUrl: frame.imageUrl,
    }));

    persistEquipment(frame, currentTitle);
  }

  function selectTitle(title) {
    const currentFrame = availableFrameOptions.find((frame) => frame.dbId === profileData.frameDbId || frame.id === profileData.frame) || availableFrameOptions[0];

    setProfileData((prev) => ({
      ...prev,
      title: title.name,
      titleDbId: title.dbId,
    }));

    persistEquipment(currentFrame, title);
  }


  useEffect(() => {
    return () => {
      if (sourcePhoto?.startsWith("blob:")) URL.revokeObjectURL(sourcePhoto);
    };
  }, [sourcePhoto]);

  function handlePhotoFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (sourcePhoto?.startsWith("blob:")) URL.revokeObjectURL(sourcePhoto);

    const url = URL.createObjectURL(file);
    setSourcePhoto(url);
    setCrop({ x: 0, y: 0, zoom: 1 });
    setActiveTab("photos");
  }

  function startCropDrag(event) {
    if (!sourcePhoto) return;
    event.preventDefault();
    const point = "touches" in event ? event.touches[0] : event;
    setIsDraggingCrop(true);
    setDragStart({ pointerX: point.clientX, pointerY: point.clientY, cropX: crop.x, cropY: crop.y });
  }

  function moveCropDrag(event) {
    if (!isDraggingCrop || !dragStart) return;
    const point = "touches" in event ? event.touches[0] : event;
    const nextX = dragStart.cropX + point.clientX - dragStart.pointerX;
    const nextY = dragStart.cropY + point.clientY - dragStart.pointerY;
    setCrop((prev) => ({ ...prev, x: nextX, y: nextY }));
  }

  function endCropDrag() {
    setIsDraggingCrop(false);
    setDragStart(null);
  }

  function saveCroppedPhoto() {
    if (!sourcePhoto) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const outputSize = 512;
      const previewSize = 280;
      const canvas = document.createElement("canvas");
      canvas.width = outputSize;
      canvas.height = outputSize;
      const ctx = canvas.getContext("2d");

      ctx.clearRect(0, 0, outputSize, outputSize);
      ctx.save();
      ctx.beginPath();
      ctx.arc(outputSize / 2, outputSize / 2, outputSize / 2, 0, Math.PI * 2);
      ctx.clip();

      const baseScale = Math.max(previewSize / img.width, previewSize / img.height);
      const scale = baseScale * crop.zoom * (outputSize / previewSize);
      const drawWidth = img.width * scale;
      const drawHeight = img.height * scale;
      const drawX = outputSize / 2 - drawWidth / 2 + crop.x * (outputSize / previewSize);
      const drawY = outputSize / 2 - drawHeight / 2 + crop.y * (outputSize / previewSize);

      ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
      ctx.restore();

      persistPhoto(canvas.toDataURL("image/png"));
    };
    img.src = sourcePhoto;
  }

  return (
    <div className="modal-overlay">
      <div className="identity-customizer modal-in">
        <div className="customizer-left">
          <div className="customizer-preview-card">
            <IdentityFrame
              photo={profileData.photo}
              frame={selectedFrame}
              size="xlarge"
            />

            <div className="customizer-preview-meta">
              <h3>{profileData.name}</h3>
              <p>{profileData.title}</p>
            </div>
          </div>
        </div>

        <div className="customizer-right">
          <div className="customizer-header">
            <h2>Personalizar identidad</h2>
            <button className="close-button" onClick={onClose}>×</button>
          </div>

          <div className="customizer-tabs">
            <button
              className={activeTab === "frames" ? "active" : ""}
              onClick={() => setActiveTab("frames")}
            >
              Marcos
            </button>
            <button
              className={activeTab === "titles" ? "active" : ""}
              onClick={() => setActiveTab("titles")}
            >
              Títulos
            </button>
            <button
              className={activeTab === "photos" ? "active" : ""}
              onClick={() => setActiveTab("photos")}
            >
              Foto
            </button>
          </div>

          {activeTab === "frames" && (
            <div className="customizer-grid frame-grid">
              {availableFrameOptions.length === 0 && (
                <p className="empty-objectives-message">No se pudieron cargar los marcos disponibles.</p>
              )}
              {availableFrameOptions.map((frame) => (
                <button
                  key={frame.id}
                  className={`frame-option-card ${profileData.frame === frame.id ? "selected" : ""}`}
                  onClick={() => selectFrame(frame)}
                  disabled={frame.disponible === false}
                  title={frame.disponible === false ? "Aún no disponible" : frame.name}
                >
                  <FrameOnlyPreview frame={frame} />
                  <strong>{frame.name}</strong>
                </button>
              ))}
            </div>
          )}

          {activeTab === "titles" && (
            <div className="customizer-grid title-grid">
              {availableTitleOptions.length === 0 && (
                <p className="empty-objectives-message">No se pudieron cargar los títulos disponibles.</p>
              )}
              {availableTitleOptions.map((title) => (
                <button
                  key={title.id}
                  className={`title-option-card ${profileData.title === title.name ? "selected" : ""}`}
                  onClick={() => selectTitle(title)}
                  disabled={title.disponible === false}
                  title={title.disponible === false ? "Aún no disponible" : title.name}
                >
                  <strong>{title.name}</strong>
                  <span>{title.level}</span>
                </button>
              ))}
            </div>
          )}

          {activeTab === "photos" && (
            <div className="customizer-grid photos-grid photo-upload-panel">
              <div className="photo-upload-actions">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handlePhotoFile}
                  hidden
                />
                <button className="primary" type="button" onClick={() => fileInputRef.current?.click()}>
                  Cargar foto
                </button>
                <p>
                  Elegí una imagen y movela dentro del círculo. La app guardará solo el recorte circular,
                  como una foto de perfil clásica.
                </p>
              </div>

              {sourcePhoto && (
                <div className="cropper-card">
                  <div
                    className="profile-cropper"
                    onMouseDown={startCropDrag}
                    onMouseMove={moveCropDrag}
                    onMouseUp={endCropDrag}
                    onMouseLeave={endCropDrag}
                    onTouchStart={startCropDrag}
                    onTouchMove={moveCropDrag}
                    onTouchEnd={endCropDrag}
                  >
                    <img
                      src={sourcePhoto}
                      alt="Recorte de foto"
                      style={{
                        transform: `translate(calc(-50% + ${crop.x}px), calc(-50% + ${crop.y}px)) scale(${crop.zoom})`,
                      }}
                      draggable={false}
                    />
                    <div className="cropper-mask" />
                    <div className="cropper-circle" />
                  </div>

                  <label className="zoom-control">
                    <span>Zoom</span>
                    <input
                      type="range"
                      min="1"
                      max="2.6"
                      step="0.05"
                      value={crop.zoom}
                      onChange={(event) => setCrop((prev) => ({ ...prev, zoom: Number(event.target.value) }))}
                    />
                  </label>

                  <button className="secondary" type="button" onClick={saveCroppedPhoto}>
                    Usar esta foto
                  </button>
                </div>
              )}

              <div className="default-photo-list">
                <h4>Avatares rápidos</h4>
                <div className="default-photo-grid">
                  {PROFILE_PHOTOS.map((photo) => (
                    <button
                      key={photo.id}
                      className={`photo-option ${profileData.photo === photo.label ? "selected" : ""}`}
                      onClick={() => persistPhoto(photo.label)}
                    >
                      {photo.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FrameOnlyPreview({ frame }) {
  return (
    <div className="frame-only-preview">
      {frame?.imageUrl ? (
        <img src={frame.imageUrl} alt={frame.name} />
      ) : (
        <span />
      )}
    </div>
  );
}

function StatsScreen({ onBack, usuarioId, isGuest = false }) {
  return (
    <div className="page-stack">
      <BackButton onBack={onBack} />

      <div className="screen-top card fade-up">
        <div>
          <h2>Estadísticas</h2>
          <p className="screen-description">
            Resumen del desempeño registrado durante las prácticas con cámara.
          </p>
        </div>
      </div>

      <PracticeProgressPanel usuarioId={usuarioId} isGuest={isGuest} />
    </div>
  );
}

function AchievementIcon({ achievement, className = "" }) {
  const unlocked = Boolean(achievement.unlocked ?? achievement.active);

  return (
    <div className={`achievement-image-wrap ${unlocked ? "unlocked" : "locked"} ${className}`}>
      {achievement.imageUrl ? (
        <img
          src={achievement.imageUrl}
          alt={achievement.name}
          onError={(event) => {
            event.currentTarget.style.display = "none";
          }}
        />
      ) : (
        <span>{unlocked ? "🏅" : "🔒"}</span>
      )}
    </div>
  );
}

function Badge({ achievement, showDescription = false }) {
  const unlocked = Boolean(achievement.unlocked ?? achievement.active);

  return (
    <div className={`badge ${unlocked ? "active" : "locked"}`}>
      <AchievementIcon achievement={achievement} />
      <strong>{achievement.name}</strong>
      {showDescription && <span>{achievement.description}</span>}
    </div>
  );
}

function GamificationToast({ event, onClose }) {
  const isLevel = event.tipo === "nivel";
  const isObjective = event.tipo === "objetivo";
  const isAchievement = event.tipo === "logro";

  const label = isLevel
    ? "Nuevo nivel"
    : isAchievement
      ? "Logro desbloqueado"
      : isObjective
        ? "Objetivo completado"
        : "Experiencia ganada";

  const title = isLevel
    ? `Nivel ${event.nivel_nuevo}`
    : isAchievement
      ? event.logro_nombre || "Nuevo logro"
      : `+${event.xp || 0} XP`;

  const detail = isAchievement
    ? event.logro_descripcion || event.mensaje
    : isObjective
      ? event.objetivo_titulo || event.mensaje
      : event.mensaje;

  const icon = isLevel
    ? "🚀"
    : isAchievement
      ? "🏅"
      : isObjective
        ? "🎯"
        : "✨";

  const toastClass = isLevel
    ? "level-up"
    : isAchievement
      ? "achievement-unlocked"
      : isObjective
        ? "objective-completed"
        : "xp-gain";

  return (
    <div className={`gamification-toast ${toastClass}`}>
      <div className="gamification-toast-icon">
        <GamificationToastIcon event={event} fallbackIcon={icon} />
      </div>

      <div className="gamification-toast-content">
        <small>{label}</small>
        <strong>{title}</strong>
        <span>{detail}</span>

        {isObjective && event.tipo_periodo && (
          <em>
            {event.tipo_periodo === "diario" ? "Objetivo diario" : "Objetivo semanal"}
          </em>
        )}

        {isAchievement && event.logro_familia && (
          <em>{event.logro_familia}</em>
        )}
      </div>

      <button className="toast-close" onClick={onClose}>×</button>
    </div>
  );
}

function GamificationToastIcon({ event, fallbackIcon }) {
  const [imageError, setImageError] = useState(false);

  if (
    event.tipo === "logro" &&
    event.logro_imagen_url &&
    !imageError
  ) {
    return (
      <img
        src={event.logro_imagen_url}
        alt={event.logro_nombre || "Logro desbloqueado"}
        onError={() => setImageError(true)}
      />
    );
  }

  return <span>{fallbackIcon}</span>;
}

function Stat({ label, value }) {
  return (
    <div className="stat-box">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProgressBar({ current, total }) {
  return (
    <div className="progress">
      <div style={{ width: `${(current / total) * 100}%` }} />
    </div>
  );
}

function BackButton({ onBack }) {
  return (
    <button className="back-button" onClick={onBack}>
      ← Volver
    </button>
  );
}