#model_gru.py
from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    Clasificador GRU multiclase para secuencias de landmarks.

    Entrada esperada:
        x.shape = (batch_size, frames_per_video, features_per_frame)

    Ejemplo:
        x.shape = (32, 20, 93)

    Salida:
        logits.shape = (batch_size, num_classes)

    Los logits luego se convierten en probabilidades con softmax durante evaluación,
    pero para entrenamiento se usan directamente con CrossEntropyLoss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Si num_layers=1, PyTorch ignora dropout dentro de GRU.
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * direction_factor),
            nn.Linear(hidden_dim * direction_factor, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Ejecuta la red.

        Para clasificación usamos el último estado temporal de la GRU.
        """
        output, hidden = self.gru(x)

        # output contiene todos los estados temporales:
        # output.shape = (batch, frames, hidden_dim * directions)
        last_output = output[:, -1, :]

        logits = self.classifier(last_output)
        return logits