from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent

IMAGES_ROOT = ROOT / "public" / "assets" / "signs"
VIDEOS_ROOT = ROOT / "public" / "assets" / "videos"

OUTPUT_DIR = ROOT / "reportes"
OUTPUT_CSV = OUTPUT_DIR / "media_audit.csv"


def mb(size_bytes: int) -> float:
    return size_bytes / (1024 * 1024)


def audit_image(path: Path) -> dict:
    image = cv2.imread(str(path))

    if image is None:
        return {
            "tipo": "imagen",
            "categoria": path.parent.name,
            "archivo": path.name,
            "ruta": str(path.relative_to(ROOT)),
            "tamano_mb": round(mb(path.stat().st_size), 3),
            "ancho": None,
            "alto": None,
            "fps": None,
            "duracion_s": None,
            "bitrate_mbps_estimado": None,
            "codec": None,
            "error": "OpenCV no pudo abrir la imagen",
        }

    height, width = image.shape[:2]

    return {
        "tipo": "imagen",
        "categoria": path.parent.name,
        "archivo": path.name,
        "ruta": str(path.relative_to(ROOT)),
        "tamano_mb": round(mb(path.stat().st_size), 3),
        "ancho": int(width),
        "alto": int(height),
        "fps": None,
        "duracion_s": None,
        "bitrate_mbps_estimado": None,
        "codec": None,
        "error": None,
    }


def decode_fourcc(value: float) -> str:
    value = int(value)

    chars = [
        chr((value >> (8 * i)) & 0xFF)
        for i in range(4)
    ]

    return "".join(chars).strip()


def audit_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        return {
            "tipo": "video",
            "categoria": path.parent.name,
            "archivo": path.name,
            "ruta": str(path.relative_to(ROOT)),
            "tamano_mb": round(mb(path.stat().st_size), 3),
            "ancho": None,
            "alto": None,
            "fps": None,
            "duracion_s": None,
            "bitrate_mbps_estimado": None,
            "codec": None,
            "error": "OpenCV no pudo abrir el video",
        }

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))

    capture.release()

    duration = frame_count / fps if fps > 0 else 0
    size_bytes = path.stat().st_size

    bitrate_mbps = (
        (size_bytes * 8) / duration / 1_000_000
        if duration > 0
        else 0
    )

    return {
        "tipo": "video",
        "categoria": path.parent.name,
        "archivo": path.name,
        "ruta": str(path.relative_to(ROOT)),
        "tamano_mb": round(mb(size_bytes), 3),
        "ancho": width,
        "alto": height,
        "fps": round(fps, 2),
        "duracion_s": round(duration, 2),
        "bitrate_mbps_estimado": round(bitrate_mbps, 3),
        "codec": fourcc,
        "error": None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    if IMAGES_ROOT.exists():
        image_paths = sorted(
            path
            for path in IMAGES_ROOT.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".webp"}
        )

        for path in image_paths:
            rows.append(audit_image(path))

    if VIDEOS_ROOT.exists():
        for path in sorted(VIDEOS_ROOT.rglob("*.mp4")):
            rows.append(audit_video(path))

    columns = [
        "tipo",
        "categoria",
        "archivo",
        "ruta",
        "tamano_mb",
        "ancho",
        "alto",
        "fps",
        "duracion_s",
        "bitrate_mbps_estimado",
        "codec",
        "error",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = defaultdict(
        lambda: {
            "cantidad": 0,
            "tamano_mb": 0.0,
        }
    )

    for row in rows:
        key = (row["tipo"], row["categoria"])
        summary[key]["cantidad"] += 1
        summary[key]["tamano_mb"] += row["tamano_mb"]

    print()
    print("=" * 72)
    print("AUDITORÍA DE RECURSOS MULTIMEDIA DE SEÑAPP")
    print("=" * 72)

    for (tipo, categoria), values in sorted(summary.items()):
        print(
            f"{tipo:8} | "
            f"{categoria:15} | "
            f"{values['cantidad']:3} archivos | "
            f"{values['tamano_mb']:8.2f} MB"
        )

    print("-" * 72)

    images = [row for row in rows if row["tipo"] == "imagen"]
    videos = [row for row in rows if row["tipo"] == "video"]

    print(
        f"Imágenes: {len(images)} archivos, "
        f"{sum(x['tamano_mb'] for x in images):.2f} MB"
    )

    print(
        f"Videos:   {len(videos)} archivos, "
        f"{sum(x['tamano_mb'] for x in videos):.2f} MB"
    )

    print(
        f"TOTAL:    {len(rows)} archivos, "
        f"{sum(x['tamano_mb'] for x in rows):.2f} MB"
    )

    errors = [row for row in rows if row["error"]]

    if errors:
        print()
        print("ARCHIVOS CON ERROR:")
        for row in errors:
            print(f"- {row['ruta']}: {row['error']}")

    print()
    print(f"CSV generado en:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()