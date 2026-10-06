from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SOURCE_IMAGES_ROOT = ROOT / "public" / "assets" / "signs"
SOURCE_VIDEOS_ROOT = ROOT / "public" / "assets" / "videos"

OUTPUT_IMAGES_ROOT = ROOT / "public" / "assets_optimized" / "signs"
OUTPUT_VIDEOS_ROOT = ROOT / "public" / "assets_optimized" / "videos"

REPORT_DIR = ROOT / "reportes"
REPORT_CSV = REPORT_DIR / "media_optimization_report.csv"

IMAGE_MAX_WIDTH = 960
IMAGE_QUALITY = 82

VIDEO_MAX_WIDTH = 1280
VIDEO_CRF = 23
VIDEO_PRESET = "slow"


def mb(size_bytes: int) -> float:
    return size_bytes / (1024 * 1024)


def run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print()
        print("ERROR EJECUTANDO:")
        print(" ".join(command))
        print()
        print(result.stderr)
        raise RuntimeError("FFmpeg devolvió un error.")


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("No se encontró ffmpeg en PATH.")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("No se encontró ffprobe en PATH.")


def optimize_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",

        "-i",
        str(source),

        "-vf",
        f"scale='min({IMAGE_MAX_WIDTH},iw)':-2",

        "-frames:v",
        "1",

        "-c:v",
        "libwebp",

        "-quality",
        str(IMAGE_QUALITY),

        "-compression_level",
        "6",

        "-preset",
        "picture",

        str(destination),
    ])


def optimize_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",

        "-i",
        str(source),

        "-map",
        "0:v:0",

        "-vf",
        f"scale='min({VIDEO_MAX_WIDTH},iw)':-2",

        "-c:v",
        "libx264",

        "-preset",
        VIDEO_PRESET,

        "-crf",
        str(VIDEO_CRF),

        "-pix_fmt",
        "yuv420p",

        "-an",

        "-movflags",
        "+faststart",

        str(destination),
    ])


def main() -> None:
    check_ffmpeg()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    image_sources = sorted(
        SOURCE_IMAGES_ROOT.rglob("*.png")
    )

    video_sources = sorted(
        SOURCE_VIDEOS_ROOT.rglob("*.mp4")
    )

    print()
    print("=" * 80)
    print("OPTIMIZACIÓN COMPLETA DE RECURSOS MULTIMEDIA DE SEÑAPP")
    print("=" * 80)
    print()
    print(f"Imágenes: {len(image_sources)}")
    print(f"Videos:   {len(video_sources)}")
    print()

    report_rows = []

    image_original_total = 0
    image_optimized_total = 0

    # ============================================================
    # IMÁGENES
    # ============================================================

    print("=" * 80)
    print("IMÁGENES")
    print("=" * 80)

    for index, source in enumerate(image_sources, start=1):

        relative = source.relative_to(SOURCE_IMAGES_ROOT)

        destination = (
            OUTPUT_IMAGES_ROOT
            / relative.parent
            / f"{source.stem}.webp"
        )

        print(
            f"[IMG {index:03}/{len(image_sources):03}] "
            f"{relative}"
        )

        optimize_image(source, destination)

        original_size = source.stat().st_size
        optimized_size = destination.stat().st_size

        image_original_total += original_size
        image_optimized_total += optimized_size

        reduction = (
            100 * (1 - optimized_size / original_size)
            if original_size
            else 0
        )

        print(
            f"    {mb(original_size):7.3f} MB"
            f" -> {mb(optimized_size):7.3f} MB"
            f" | reducción {reduction:5.1f}%"
        )

        report_rows.append({
            "tipo": "imagen",
            "categoria": relative.parent.as_posix(),
            "archivo_original": source.name,
            "archivo_optimizado": destination.name,
            "original_mb": round(mb(original_size), 4),
            "optimizado_mb": round(mb(optimized_size), 4),
            "reduccion_pct": round(reduction, 2),
        })

    print()

    # ============================================================
    # VIDEOS
    # ============================================================

    print("=" * 80)
    print("VIDEOS")
    print("=" * 80)

    video_original_total = 0
    video_optimized_total = 0

    for index, source in enumerate(video_sources, start=1):

        relative = source.relative_to(SOURCE_VIDEOS_ROOT)

        destination = (
            OUTPUT_VIDEOS_ROOT
            / relative
        )

        print(
            f"[VID {index:03}/{len(video_sources):03}] "
            f"{relative}"
        )

        optimize_video(source, destination)

        original_size = source.stat().st_size
        optimized_size = destination.stat().st_size

        video_original_total += original_size
        video_optimized_total += optimized_size

        reduction = (
            100 * (1 - optimized_size / original_size)
            if original_size
            else 0
        )

        print(
            f"    {mb(original_size):7.3f} MB"
            f" -> {mb(optimized_size):7.3f} MB"
            f" | reducción {reduction:5.1f}%"
        )

        report_rows.append({
            "tipo": "video",
            "categoria": relative.parent.as_posix(),
            "archivo_original": source.name,
            "archivo_optimizado": destination.name,
            "original_mb": round(mb(original_size), 4),
            "optimizado_mb": round(mb(optimized_size), 4),
            "reduccion_pct": round(reduction, 2),
        })

    # ============================================================
    # CSV
    # ============================================================

    with REPORT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        columns = [
            "tipo",
            "categoria",
            "archivo_original",
            "archivo_optimizado",
            "original_mb",
            "optimizado_mb",
            "reduccion_pct",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    # ============================================================
    # RESUMEN
    # ============================================================

    original_total = (
        image_original_total
        + video_original_total
    )

    optimized_total = (
        image_optimized_total
        + video_optimized_total
    )

    total_reduction = (
        100 * (1 - optimized_total / original_total)
        if original_total
        else 0
    )

    image_reduction = (
        100 * (
            1 - image_optimized_total / image_original_total
        )
        if image_original_total
        else 0
    )

    video_reduction = (
        100 * (
            1 - video_optimized_total / video_original_total
        )
        if video_original_total
        else 0
    )

    print()
    print("=" * 80)
    print("RESULTADO FINAL")
    print("=" * 80)

    print()
    print("IMÁGENES")
    print(
        f"Original:   {mb(image_original_total):8.2f} MB"
    )
    print(
        f"Optimizado: {mb(image_optimized_total):8.2f} MB"
    )
    print(
        f"Reducción:  {image_reduction:8.1f}%"
    )

    print()
    print("VIDEOS")
    print(
        f"Original:   {mb(video_original_total):8.2f} MB"
    )
    print(
        f"Optimizado: {mb(video_optimized_total):8.2f} MB"
    )
    print(
        f"Reducción:  {video_reduction:8.1f}%"
    )

    print()
    print("TOTAL")
    print(
        f"Original:   {mb(original_total):8.2f} MB"
    )
    print(
        f"Optimizado: {mb(optimized_total):8.2f} MB"
    )
    print(
        f"Reducción:  {total_reduction:8.1f}%"
    )

    print()
    print("Reporte:")
    print(REPORT_CSV)

    print()
    print("Recursos optimizados:")
    print(OUTPUT_IMAGES_ROOT)
    print(OUTPUT_VIDEOS_ROOT)
    print()


if __name__ == "__main__":
    main()