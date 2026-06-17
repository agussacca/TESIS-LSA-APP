# core/model_gru.py
from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    Clasificador GRU multiclase para secuencias de landmarks.

    Entrada esperada:
        x.shape = (batch_size, frames_per_video, features_per_frame)

    Ejemplo:
        x.shape = (32, 20, 121)

    Salida:
        logits.shape = (batch_size, num_classes)

    Los logits luego se convierten en probabilidades con softmax durante
    evaluación o inferencia, pero para entrenamiento se usan directamente
    con CrossEntropyLoss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        classifier_hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.classifier_hidden_dim = classifier_hidden_dim

        # En PyTorch, dropout dentro de GRU solo se aplica entre capas
        # recurrentes. Si num_layers=1, no tiene efecto dentro de la GRU.
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        recurrent_output_dim = hidden_dim * direction_factor

        self.classifier = nn.Sequential(
            nn.LayerNorm(recurrent_output_dim),
            nn.Linear(recurrent_output_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta la red.

        Para clasificación se utiliza el último estado temporal de la GRU.
        """
        output, _ = self.gru(x)

        # output.shape = (batch, frames, hidden_dim * directions)
        last_output = output[:, -1, :]

        logits = self.classifier(last_output)
        return logits


class LSTMClassifier(nn.Module):
    """
    Clasificador LSTM multiclase para secuencias de landmarks.

    Se mantiene una interfaz equivalente a GRUClassifier para poder comparar
    modelos candidatos bajo el mismo pipeline experimental.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        classifier_hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.classifier_hidden_dim = classifier_hidden_dim

        # En PyTorch, dropout dentro de LSTM solo se aplica entre capas
        # recurrentes. Si num_layers=1, no tiene efecto dentro de la LSTM.
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        recurrent_output_dim = hidden_dim * direction_factor

        self.classifier = nn.Sequential(
            nn.LayerNorm(recurrent_output_dim),
            nn.Linear(recurrent_output_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta la red.

        Para clasificación se utiliza el último estado temporal de la LSTM.
        """
        output, _ = self.lstm(x)

        # output.shape = (batch, frames, hidden_dim * directions)
        last_output = output[:, -1, :]

        logits = self.classifier(last_output)
        return logits