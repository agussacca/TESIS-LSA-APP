from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SOURCE_IMAGES = ROOT / "public" / "assets" / "signs" / "abecedario"
SOURCE_VIDEOS = ROOT / "public" / "assets" / "videos" / "abecedario"

OUTPUT_IMAGES = (
    ROOT / "public" / "assets_optimized" / "signs" / "abecedario"
)
OUTPUT_VIDEOS = (
    ROOT / "public" / "assets_optimized" / "videos" / "abecedario"
)

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
        raise RuntimeError(
            "No se encontró ffmpeg en PATH."
        )

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "No se encontró ffprobe en PATH."
        )


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

        # Sólo procesamos el stream de video.
        "-map",
        "0:v:0",

        # 1920x1080 -> máximo 1280x720.
        # No se aumenta la resolución si el original fuera menor.
        "-vf",
        f"scale='min({VIDEO_MAX_WIDTH},iw)':-2",

        # H.264 compatible con navegadores.
        "-c:v",
        "libx264",

        "-preset",
        VIDEO_PRESET,

        "-crf",
        str(VIDEO_CRF),

        "-pix_fmt",
        "yuv420p",

        # La aplicación reproduce los videos silenciados.
        "-an",

        # Mueve la metadata MP4 al comienzo para reproducción web.
        "-movflags",
        "+faststart",

        str(destination),
    ])


def main() -> None:
    check_ffmpeg()

    if not SOURCE_IMAGES.exists():
        raise RuntimeError(
            f"No existe: {SOURCE_IMAGES}"
        )

    if not SOURCE_VIDEOS.exists():
        raise RuntimeError(
            f"No existe: {SOURCE_VIDEOS}"
        )

    image_sources = sorted(SOURCE_IMAGES.glob("*.png"))
    video_sources = sorted(SOURCE_VIDEOS.glob("*.mp4"))

    print()
    print("=" * 76)
    print("OPTIMIZACIÓN PILOTO - ABECEDARIO")
    print("=" * 76)
    print()
    print(f"Imágenes encontradas: {len(image_sources)}")
    print(f"Videos encontrados:   {len(video_sources)}")
    print()

    image_original_total = 0
    image_optimized_total = 0

    for index, source in enumerate(image_sources, start=1):
        destination = OUTPUT_IMAGES / f"{source.stem}.webp"

        print(
            f"[IMG {index:02}/{len(image_sources):02}] "
            f"{source.name} -> {destination.name}"
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
            f"    {mb(original_size):.2f} MB"
            f" -> {mb(optimized_size):.2f} MB"
            f" | reducción {reduction:.1f}%"
        )

    print()
    print("-" * 76)
    print()

    video_original_total = 0
    video_optimized_total = 0

    for index, source in enumerate(video_sources, start=1):
        destination = OUTPUT_VIDEOS / source.name

        print(
            f"[VID {index:02}/{len(video_sources):02}] "
            f"{source.name}"
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
            f"    {mb(original_size):.2f} MB"
            f" -> {mb(optimized_size):.2f} MB"
            f" | reducción {reduction:.1f}%"
        )

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

    print()
    print("=" * 76)
    print("RESULTADO")
    print("=" * 76)

    print()
    print("IMÁGENES")
    print(
        f"Original:   {mb(image_original_total):8.2f} MB"
    )
    print(
        f"Optimizado: {mb(image_optimized_total):8.2f} MB"
    )

    print()
    print("VIDEOS")
    print(
        f"Original:   {mb(video_original_total):8.2f} MB"
    )
    print(
        f"Optimizado: {mb(video_optimized_total):8.2f} MB"
    )

    print()
    print("TOTAL ABECEDARIO")
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
    print("Archivos generados en:")
    print(OUTPUT_IMAGES)
    print(OUTPUT_VIDEOS)
    print()


if __name__ == "__main__":
    main()