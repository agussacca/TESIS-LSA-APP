from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from config import (
    CATEGORIES_JSON,
    DATASET_DIR,
    FRAMES_PER_VIDEO,
    FEATURES_PER_FRAME,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Piloto inicial recomendado:
# - A: una mano estática simple
# - E: una mano con referencia a cara
# - L: una mano estática clara
# - J: dinámica
# - Ñ: dos manos
PILOT_LABELS = ["A", "E", "L", "J", "Ñ"]


@dataclass(frozen=True)
class Category:
    """
    Representa una clase/letra del abecedario.

    Esta estructura se carga desde dataset/metadata/categories_abecedario.json
    y se usa tanto para grabar como para extraer features.
    """

    label: str
    hands: int
    motion: str
    count_per_subject: int


@dataclass(frozen=True)
class VideoSample:
    """
    Representa un video individual del dataset.

    Ejemplo de nombre esperado:
        A_001_001.mp4

    Donde:
        A   -> label
        001 -> subject_id
        001 -> take_id
    """

    label: str
    subject_id: str
    take_id: str
    video_path: Path
    hands: int
    motion: str


def load_categories(categories_path: Path = CATEGORIES_JSON) -> list[Category]:
    """
    Lee el archivo JSON de categorías y devuelve una lista de Category.

    Valida que cada categoría tenga:
    - label
    - hands
    - motion
    - count_per_subject
    """

    if not categories_path.exists():
        raise FileNotFoundError(f"No existe el archivo de categorías: {categories_path}")

    with open(categories_path, "r", encoding="utf-8") as f:
        raw_categories = json.load(f)

    categories = []

    for item in raw_categories:
        required_keys = {"label", "hands", "motion", "count_per_subject"}
        missing = required_keys - set(item.keys())

        if missing:
            raise ValueError(
                f"Categoría inválida. Faltan campos {missing}. Item: {item}"
            )

        label = str(item["label"]).strip()
        hands = int(item["hands"])
        motion = str(item["motion"]).strip().lower()
        count_per_subject = int(item["count_per_subject"])

        if hands not in (1, 2):
            raise ValueError(f"hands debe ser 1 o 2 para label={label}")

        if motion not in ("static", "dynamic"):
            raise ValueError(f"motion debe ser static/dynamic para label={label}")

        categories.append(
            Category(
                label=label,
                hands=hands,
                motion=motion,
                count_per_subject=count_per_subject,
            )
        )

    return categories


def get_category_map(categories: list[Category]) -> dict[str, Category]:
    """
    Convierte la lista de categorías en un diccionario por label.

    Ejemplo:
        category_map["A"] -> Category(label="A", hands=1, ...)
    """

    return {cat.label: cat for cat in categories}


def filter_categories(
    categories: list[Category],
    pilot: bool = False,
    labels: list[str] | None = None,
) -> list[Category]:
    """
    Filtra categorías según el modo de grabación/extracción.

    - pilot=True usa PILOT_LABELS.
    - labels=["A", "B"] permite elegir manualmente.
    - sin filtros devuelve todas.
    """

    if pilot and labels:
        raise ValueError("No usar pilot=True y labels al mismo tiempo.")

    if pilot:
        selected = set(PILOT_LABELS)
        return [cat for cat in categories if cat.label in selected]

    if labels:
        selected = set(labels)
        return [cat for cat in categories if cat.label in selected]

    return categories


def ensure_dataset_folders(
    categories: list[Category],
    dataset_dir: Path = DATASET_DIR,
) -> None:
    """
    Crea una carpeta por letra dentro del dataset.

    Ejemplo:
        dataset/abecedario_lsa_raw/A
        dataset/abecedario_lsa_raw/Ñ
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories:
        (dataset_dir / cat.label).mkdir(parents=True, exist_ok=True)


def parse_video_filename(video_path: Path) -> tuple[str | None, str | None, str | None]:
    """
    Parsea nombres con formato:
        LETRA_SUBJECT_TAKE.mp4

    Ejemplos válidos:
        A_001_001.mp4
        Ñ_001_001.mp4

    Retorna:
        label, subject_id, take_id

    Si el nombre no cumple el formato, retorna None, None, None.
    """

    stem = video_path.stem
    parts = stem.split("_")

    if len(parts) != 3:
        return None, None, None

    label, subject_id, take_id = parts
    return label, subject_id, take_id


def get_next_take_id(category_dir: Path, label: str, subject_id: str) -> int:
    """
    Busca el siguiente número de toma para una letra y sujeto.

    Si ya existen:
        A_001_001.mp4
        A_001_002.mp4

    devuelve:
        3
    """

    category_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{label}_{subject_id}_*.mp4"
    existing_files = list(category_dir.glob(pattern))

    max_take = 0

    for path in existing_files:
        parsed_label, parsed_subject, parsed_take = parse_video_filename(path)

        if parsed_label != label or parsed_subject != subject_id:
            continue

        try:
            take_number = int(parsed_take)
            max_take = max(max_take, take_number)
        except (TypeError, ValueError):
            continue

    return max_take + 1


def list_video_samples(
    dataset_dir: Path = DATASET_DIR,
    categories_path: Path = CATEGORIES_JSON,
) -> list[VideoSample]:
    """
    Recorre el dataset y devuelve una lista de videos válidos.

    Valida que:
    - la carpeta corresponda a una categoría conocida;
    - el nombre del archivo tenga formato correcto;
    - la extensión sea de video.
    """

    categories = load_categories(categories_path)
    category_map = get_category_map(categories)

    samples = []

    if not dataset_dir.exists():
        return samples

    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue

        folder_label = class_dir.name

        if folder_label not in category_map:
            # Carpeta no reconocida; se ignora.
            continue

        category = category_map[folder_label]

        for video_path in class_dir.iterdir():
            if not video_path.is_file():
                continue

            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            label, subject_id, take_id = parse_video_filename(video_path)

            if label is None:
                continue

            # El label del nombre debe coincidir con la carpeta.
            if label != folder_label:
                continue

            samples.append(
                VideoSample(
                    label=label,
                    subject_id=subject_id,
                    take_id=take_id,
                    video_path=video_path,
                    hands=category.hands,
                    motion=category.motion,
                )
            )

    return samples


def build_dataset_header() -> list[str]:
    """
    Construye el header del CSV principal de landmarks.

    Columnas iniciales:
        label, video_name, subject_id, take_id, hands, motion

    Luego agrega:
        f0_0 ... f0_92
        f1_0 ... f1_92
        ...
        f19_92

    Total de features:
        FRAMES_PER_VIDEO * FEATURES_PER_FRAME
    """

    header = [
        "label",
        "video_name",
        "subject_id",
        "take_id",
        "hands",
        "motion",
    ]

    for frame_idx in range(FRAMES_PER_VIDEO):
        for feature_idx in range(FEATURES_PER_FRAME):
            header.append(f"f{frame_idx}_{feature_idx}")

    return header


def build_stats_header() -> list[str]:
    """
    Header del CSV de estadísticas de extracción.

    Estas métricas permiten auditar si un video fue procesado correctamente.
    """

    return [
        "label",
        "video_name",
        "subject_id",
        "take_id",
        "hands",
        "motion",
        "total_frames_video",
        "sampled_frames_requested",
        "sampled_frames_read_ok",
        "frames_with_any_hand",
        "frames_with_expected_hands",
        "detection_ratio_any",
        "detection_ratio_expected",
        "avg_detected_hands",
        "avg_used_hands",
        "zero_ratio",
        "elapsed_seconds",
    ]