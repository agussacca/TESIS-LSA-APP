#hand_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    MEDIAPIPE_HAND_MODEL_PATH,
    MEDIAPIPE_POSE_MODEL_PATH,
    MAX_NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIN_POSE_DETECTION_CONFIDENCE,
    MIN_POSE_PRESENCE_CONFIDENCE,
    MIN_POSE_TRACKING_CONFIDENCE,
)


@dataclass
class HandPoseResult:
    """
    Resultado combinado de MediaPipe para un frame.

    hand_result:
        Resultado de HandLandmarker.

    pose_result:
        Resultado de PoseLandmarker.
    """

    hand_result: Any
    pose_result: Any


def _frame_bgr_to_mp_image(frame_bgr) -> mp.Image:
    """
    Convierte un frame BGR de OpenCV a mp.Image RGB/SRGB.

    OpenCV trabaja en BGR.
    MediaPipe espera RGB/SRGB.
    """

    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )


class HandDetector:
    """
    Encapsula MediaPipe HandLandmarker.

    Este detector se mantiene compatible con V1:
      - recibe un frame BGR;
      - devuelve únicamente el resultado de manos.

    La lógica de conversión a features queda separada en core/features.py.
    """

    def __init__(self):
        if not MEDIAPIPE_HAND_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo de manos de MediaPipe: "
                f"{MEDIAPIPE_HAND_MODEL_PATH}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(MEDIAPIPE_HAND_MODEL_PATH)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect_for_video(self, frame_bgr, timestamp_ms: int):
        """
        Recibe un frame BGR de OpenCV y devuelve el resultado de MediaPipe Hands.
        """

        mp_image = _frame_bgr_to_mp_image(frame_bgr)
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        """
        Libera recursos del detector.
        """

        if self.detector is not None:
            self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PoseDetector:
    """
    Encapsula MediaPipe PoseLandmarker.

    Este detector se agrega para V2:
      - detecta anclas corporales/faciales;
      - por ahora usaremos nariz, ojos y hombros;
      - no reemplaza al detector de manos.
    """

    def __init__(self):
        if not MEDIAPIPE_POSE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo de pose de MediaPipe: "
                f"{MEDIAPIPE_POSE_MODEL_PATH}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(MEDIAPIPE_POSE_MODEL_PATH)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_POSE_TRACKING_CONFIDENCE,
            output_segmentation_masks=False,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect_for_video(self, frame_bgr, timestamp_ms: int):
        """
        Recibe un frame BGR de OpenCV y devuelve el resultado de MediaPipe Pose.
        """

        mp_image = _frame_bgr_to_mp_image(frame_bgr)
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        """
        Libera recursos del detector.
        """

        if self.detector is not None:
            self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class HandPoseDetector:
    """
    Detector combinado para V2.

    Ejecuta en cada frame:
      1. HandLandmarker
      2. PoseLandmarker

    Devuelve un HandPoseResult con ambos resultados.

    Esto permite que core/features.py V2 construya features usando:
      - forma de manos;
      - relación entre manos;
      - posición relativa de las manos respecto a cara/cuerpo.
    """

    def __init__(self):
        self.hand_detector = HandDetector()
        self.pose_detector = PoseDetector()

    def detect_for_video(self, frame_bgr, timestamp_ms: int) -> HandPoseResult:
        """
        Recibe un frame BGR de OpenCV y devuelve detección combinada manos + pose.
        """

        hand_result = self.hand_detector.detect_for_video(
            frame_bgr=frame_bgr,
            timestamp_ms=timestamp_ms,
        )

        pose_result = self.pose_detector.detect_for_video(
            frame_bgr=frame_bgr,
            timestamp_ms=timestamp_ms,
        )

        return HandPoseResult(
            hand_result=hand_result,
            pose_result=pose_result,
        )

    def close(self):
        """
        Libera recursos de ambos detectores.
        """

        self.hand_detector.close()
        self.pose_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()