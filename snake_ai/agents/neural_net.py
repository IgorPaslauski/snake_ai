from __future__ import annotations
from typing import List
import numpy as np


class NeuralNetwork:
    """
    Rede neural feedforward simples totalmente conectada.
    Pesos armazenados em um único vetor (para GA).
    """

    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights_shapes = [
            (layer_sizes[i], layer_sizes[i - 1]) for i in range(1, len(layer_sizes))
        ]
        self.biases_shapes = [(size, 1) for size in layer_sizes[1:]]
        self.num_params = sum(
            w * h for (w, h) in self.weights_shapes
        ) + sum(b for (b, _) in self.biases_shapes)

        # inicializa com pesos aleatórios pequenos
        self.params = np.random.randn(self.num_params).astype(np.float32) * 0.1

    def set_params(self, params: np.ndarray) -> None:
        if params.shape[0] != self.num_params:
            raise ValueError("Tamanho de vetor de pesos incompatível.")
        self.params = params.astype(np.float32)

    def get_params(self) -> np.ndarray:
        return self.params.copy()

    def _unpack_params(self):
        """Separa o vetor de params em listas de pesos e biases."""
        idx = 0
        weights = []
        biases = []
        for (rows, cols) in self.weights_shapes:
            size = rows * cols
            w = self.params[idx : idx + size].reshape(rows, cols)
            idx += size
            weights.append(w)
        for (rows, _) in self.biases_shapes:
            size = rows
            b = self.params[idx : idx + size].reshape(rows, 1)
            idx += size
            biases.append(b)
        return weights, biases

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: vetor (input_dim,)"""
        weights, biases = self._unpack_params()
        a = x.reshape(-1, 1)
        for i, (w, b) in enumerate(zip(weights, biases)):
            z = w @ a + b
            if i < len(weights) - 1:
                a = np.tanh(z)
            else:
                a = z  # camada de saída linear
        return a.flatten()
