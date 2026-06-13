#config.py
from pathlib import Path

# ============================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# DATASET
# ============================================================

DATASET_DIR = BASE_DIR / "dataset" / "abecedario_lsa_raw"
HOLDOUT_DIR = BASE_DIR / "dataset" / "abecedario_lsa_holdout"
LIVE_DIR = BASE_DIR / "dataset" / "abecedario_lsa_live"

METADATA_DIR = BASE_DIR / "dataset" / "metadata"
REFERENCES_DIR = BASE_DIR / "dataset" / "references"

CATEGORIES_JSON = METADATA_DIR / "categories_abecedario.json"
RECORDING_PROTOCOL_MD = METADATA_DIR / "recording_protocol.md"
REFERENCE_VIDEO_MD = METADATA_DIR / "reference_video.md"

# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = BASE_DIR / "outputs"

CSV_DIR = OUTPUT_DIR / "csv"
STATS_DIR = OUTPUT_DIR / "stats"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# V1 quedó congelado en la baseline.
# Para V2 usamos nombres nuevos para no pisar archivos anteriores.
OUTPUT_CSV = CSV_DIR / "abecedario_landmarks_v2.csv"
STATS_CSV = STATS_DIR / "abecedario_stats_v2.csv"

# ============================================================
# MODELOS
# ============================================================

MEDIAPIPE_MODELS_DIR = BASE_DIR / "models" / "mediapipe"

MEDIAPIPE_MODEL_PATH = MEDIAPIPE_MODELS_DIR / "hand_landmarker.task"
MEDIAPIPE_HAND_MODEL_PATH = MEDIAPIPE_MODEL_PATH

# Nuevo modelo para V2:
# Descargar/ubicar este archivo en:
#   models/mediapipe/pose_landmarker_lite.task
MEDIAPIPE_POSE_MODEL_PATH = MEDIAPIPE_MODELS_DIR / "pose_landmarker_lite.task"

TRAINED_MODELS_DIR = BASE_DIR / "models" / "trained"

# ============================================================
# MEDIAPIPE - HANDS
# ============================================================

# Para el abecedario completo necesitamos soportar letras de una y dos manos.
MAX_NUM_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.35
MIN_HAND_PRESENCE_CONFIDENCE = 0.35
MIN_TRACKING_CONFIDENCE = 0.35

# ============================================================
# MEDIAPIPE - POSE
# ============================================================

MIN_POSE_DETECTION_CONFIDENCE = 0.35
MIN_POSE_PRESENCE_CONFIDENCE = 0.35
MIN_POSE_TRACKING_CONFIDENCE = 0.35

# ============================================================
# FEATURES
# ============================================================

FRAMES_PER_VIDEO = 20

# ============================================================
# FEATURES V1 - MANOS
# ============================================================
#
# Mano primaria:
#   presence(1) + wrist_xy(2) + landmarks relativos xy(42) = 45
#
# Mano secundaria:
#   presence(1) + wrist_xy(2) + landmarks relativos xy(42) = 45
#
# Relación entre manos:
#   dx + dy + distancia = 3
#
# Total V1 = 93

HAND_VECTOR_SIZE = 45
HAND_RELATION_VECTOR_SIZE = 3
HAND_FEATURES_PER_FRAME = 93

# ============================================================
# FEATURES V2 - ANCLAS FACIALES/CORPORALES
# ============================================================
#
# Pose anchors:
#   pose_present              = 1
#   nose_x, nose_y            = 2
#   eye_center_x, eye_center_y = 2
#   shoulder_center_x/y       = 2
#   shoulder_width            = 1
#   face_to_shoulder_dy       = 1
#   anchor_scale              = 1
#
# Total anchors = 10

POSE_ANCHOR_FEATURE_SIZE = 10

# Relación de una mano con anclas:
#   dx, dy, dist mano -> nariz             = 3
#   dx, dy, dist mano -> centro ojos       = 3
#   dx, dy, dist mano -> centro hombros    = 3
#
# Total por mano = 9
# Dos manos = 18

HAND_TO_ANCHOR_FEATURE_SIZE = 9
BOTH_HANDS_TO_ANCHOR_FEATURE_SIZE = 18

# Total V2:
#   manos V1             = 93
#   anclas pose          = 10
#   relación mano-anclas = 18
#   total                = 121

FEATURES_PER_FRAME = (
    HAND_FEATURES_PER_FRAME
    + POSE_ANCHOR_FEATURE_SIZE
    + BOTH_HANDS_TO_ANCHOR_FEATURE_SIZE
)

# ============================================================
# ENTRENAMIENTO
# ============================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3