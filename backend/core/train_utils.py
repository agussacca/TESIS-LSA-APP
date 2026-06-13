#train_utils.py
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fija semillas para hacer los experimentos más reproducibles.

    En GPU puede seguir habiendo pequeñas variaciones, pero esto reduce bastante
    la aleatoriedad entre corridas.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Devuelve cuda si está disponible; caso contrario, cpu.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    metadata: dict | None = None,
) -> None:
    """
    Guarda un checkpoint con pesos del modelo y metadata auxiliar.

    Guardar metadata permite saber luego con qué clases y configuración
    fue entrenado el modelo.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }

    torch.save(checkpoint, path)


def load_checkpoint(path: Path, map_location=None) -> dict:
    """
    Carga un checkpoint de entrenamiento.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)