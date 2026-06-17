from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import (
    FEATURES_PER_FRAME,
    FRAMES_PER_VIDEO,
    TRAINED_MODELS_DIR,
)
from core.model_gru import GRUClassifier, LSTMClassifier
from core.train_utils import get_device, load_checkpoint


# ---------------------------------------------------------------------
# Configuración del modelo usado por la aplicación web
# ---------------------------------------------------------------------
# Para usar LSTM en la app, cambiá este nombre por el checkpoint LSTM final.
# Ejemplo:
# MODEL_NAME = "M5_lstm_h128_l1_d0p2_holdout_sujeto"
#
# También se puede sobrescribir mediante variables de entorno:
# LSA_MODEL_NAME
# LSA_MODEL_PATH
# LSA_MODEL_TYPE
# ---------------------------------------------------------------------

DEFAULT_MODEL_NAME = "M5_lstm_h128_l1_d0p2" #"abecedario_gru_v2"

MODEL_NAME = os.getenv("LSA_MODEL_NAME", DEFAULT_MODEL_NAME)

MODEL_PATH = Path(
    os.getenv(
        "LSA_MODEL_PATH",
        str(TRAINED_MODELS_DIR / f"{MODEL_NAME}_best.pt"),
    )
)


@dataclass(frozen=True)
class PredictionResult:
    model_loaded: bool
    model_path: str | None

    pred_label: str | None
    confidence: float | None

    top2_label: str | None
    top2_confidence: float | None
    top2_margin: float | None

    top_predictions: list[dict]
    all_probs: dict[str, float]

    error: str | None = None


class ModelRunner:
    """
    Ejecuta inferencia del modelo secuencial del abecedario LSA.

    El runner soporta modelos recurrentes GRU y LSTM, siempre que ambos
    mantengan la misma interfaz de entrada/salida:

        Entrada:
            sequence: np.ndarray shape=(FRAMES_PER_VIDEO, FEATURES_PER_FRAME)

        Salida:
            pred_label, confidence, top2_label, top2_confidence,
            top2_margin y top_predictions.

    La arquitectura concreta se determina a partir de la metadata del
    checkpoint, de variables de entorno o del nombre del archivo.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)

        self.device = None
        self.model: torch.nn.Module | None = None

        self.labels: list[str] = []
        self.idx_to_label: dict[int, str] = {}

        self.model_type: str | None = None
        self.model_loaded = False
        self.load_error: str | None = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No existe el modelo entrenado: {self.model_path}"
                )

            self.device = get_device()
            checkpoint = load_checkpoint(self.model_path, map_location=self.device)

            metadata = checkpoint.get("metadata")
            if metadata is None:
                raise KeyError(
                    "El checkpoint no contiene metadata. "
                    "Se requiere metadata para reconstruir la arquitectura del modelo."
                )

            metadata_frames = int(
                metadata.get("frames_per_video", FRAMES_PER_VIDEO)
            )
            metadata_features = int(
                metadata.get("features_per_frame", FEATURES_PER_FRAME)
            )

            if metadata_frames != FRAMES_PER_VIDEO:
                raise ValueError(
                    f"El modelo fue entrenado con frames_per_video={metadata_frames}, "
                    f"pero config.FRAMES_PER_VIDEO={FRAMES_PER_VIDEO}."
                )

            if metadata_features != FEATURES_PER_FRAME:
                raise ValueError(
                    f"El modelo fue entrenado con features_per_frame={metadata_features}, "
                    f"pero config.FEATURES_PER_FRAME={FEATURES_PER_FRAME}."
                )

            self.labels = list(metadata["labels"])

            label_to_idx = {
                str(label): int(idx)
                for label, idx in metadata["label_to_idx"].items()
            }

            self.idx_to_label = {
                idx: label
                for label, idx in label_to_idx.items()
            }

            self.model_type = self._resolve_model_type(metadata)

            self.model = self._build_model_from_metadata(metadata).to(self.device)

            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                raise KeyError(
                    "El checkpoint no contiene 'model_state_dict'."
                )

            self.model.load_state_dict(state_dict)
            self.model.eval()

            self.model_loaded = True
            self.load_error = None

        except Exception as exc:
            self.model_loaded = False
            self.load_error = str(exc)
            self.model = None

    def _resolve_model_type(self, metadata: dict) -> str:
        """
        Determina si el checkpoint debe cargarse como GRU o LSTM.

        Prioridad:
            1. Variable de entorno LSA_MODEL_TYPE.
            2. Campo metadata["model_type"].
            3. Nombre del archivo del checkpoint.
            4. Fallback a "gru" por compatibilidad con checkpoints antiguos.
        """
        env_model_type = os.getenv("LSA_MODEL_TYPE")

        if env_model_type:
            model_type = env_model_type
        else:
            model_type = metadata.get("model_type")

        if not model_type:
            stem = self.model_path.stem.lower()

            if "lstm" in stem:
                model_type = "lstm"
            elif "gru" in stem:
                model_type = "gru"
            else:
                model_type = "gru"

        model_type = str(model_type).lower().strip()

        if model_type not in {"gru", "lstm"}:
            raise ValueError(
                f"Tipo de modelo no soportado: {model_type}. "
                "Opciones válidas: 'gru' o 'lstm'."
            )

        return model_type

    def _build_model_from_metadata(self, metadata: dict) -> torch.nn.Module:
        if self.model_type is None:
            raise RuntimeError("No se resolvió el tipo de modelo.")

        model_class = {
            "gru": GRUClassifier,
            "lstm": LSTMClassifier,
        }[self.model_type]

        return model_class(
            input_dim=FEATURES_PER_FRAME,
            hidden_dim=int(metadata["hidden_dim"]),
            num_classes=len(self.labels),
            num_layers=int(metadata["num_layers"]),
            dropout=float(metadata["dropout"]),
            bidirectional=self._as_bool(metadata["bidirectional"]),
            classifier_hidden_dim=int(
                metadata.get("classifier_hidden_dim", 128)
            ),
        )

    def predict(self, sequence: np.ndarray, *, top_k: int = 5) -> PredictionResult:
        if not self.model_loaded or self.model is None:
            return PredictionResult(
                model_loaded=False,
                model_path=str(self.model_path),
                pred_label=None,
                confidence=None,
                top2_label=None,
                top2_confidence=None,
                top2_margin=None,
                top_predictions=[],
                all_probs={},
                error=self.load_error or "Modelo no cargado.",
            )

        self._validate_sequence(sequence)

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probs).item())
        pred_label = self.idx_to_label[pred_idx]
        confidence = float(probs[pred_idx].item())

        topk_count_for_margin = min(2, len(probs))
        top_values, top_indices = torch.topk(probs, k=topk_count_for_margin)

        top2_label = None
        top2_confidence = 0.0
        top2_margin = confidence

        if topk_count_for_margin >= 2:
            top2_idx = int(top_indices[1].item())
            top2_label = self.idx_to_label[top2_idx]
            top2_confidence = float(top_values[1].item())
            top2_margin = float(confidence - top2_confidence)

        all_probs = {
            self.idx_to_label[i]: float(probs[i].item())
            for i in range(len(probs))
        }

        top_items = sorted(
            all_probs.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        top_predictions = [
            {
                "label": label,
                "probability": probability,
            }
            for label, probability in top_items
        ]

        return PredictionResult(
            model_loaded=True,
            model_path=str(self.model_path),
            pred_label=pred_label,
            confidence=confidence,
            top2_label=top2_label,
            top2_confidence=top2_confidence,
            top2_margin=top2_margin,
            top_predictions=top_predictions,
            all_probs=all_probs,
            error=None,
        )

    def _validate_sequence(self, sequence: np.ndarray) -> None:
        if sequence is None:
            raise ValueError("La secuencia recibida es None.")

        if not isinstance(sequence, np.ndarray):
            raise TypeError("La secuencia debe ser un np.ndarray.")

        expected_shape = (FRAMES_PER_VIDEO, FEATURES_PER_FRAME)

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Shape de secuencia inválido. "
                f"Esperado={expected_shape}, recibido={sequence.shape}."
            )

        if not np.all(np.isfinite(sequence)):
            raise ValueError("La secuencia contiene NaN o infinitos.")

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            return value.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "si",
                "sí",
            }

        return bool(value)