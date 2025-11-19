from __future__ import annotations
from typing import List
import numpy as np


class NeuralNetwork:
    """
    Rede neural feedforward simples totalmente conectada.
    Pesos armazenados em um único vetor (para GA).
    """

    def __init__(self, layer_sizes: List[int], init_method: str = "xavier"):
        self.layer_sizes = layer_sizes
        self.weights_shapes = [
            (layer_sizes[i], layer_sizes[i - 1]) for i in range(1, len(layer_sizes))
        ]
        self.biases_shapes = [(size, 1) for size in layer_sizes[1:]]
        self.num_params = sum(
            w * h for (w, h) in self.weights_shapes
        ) + sum(b for (b, _) in self.biases_shapes)

        # Cache para unpack_params (inicializado como None)
        self._cached_weights = None
        self._cached_biases = None

        # Inicialização melhorada: Xavier para tanh, He para ReLU
        if init_method == "xavier":
            # Xavier/Glorot initialization - melhor para tanh
            self.params = self._xavier_init()
        elif init_method == "he":
            # He initialization - melhor para ReLU
            self.params = self._he_init()
        else:
            # Inicialização padrão (pequenos valores aleatórios)
            self.params = np.random.randn(self.num_params).astype(np.float32) * 0.1
    
    def _xavier_init(self) -> np.ndarray:
        """Inicialização Xavier/Glorot - adequada para tanh."""
        params = np.zeros(self.num_params, dtype=np.float32)
        idx = 0
        
        for i, (rows, cols) in enumerate(self.weights_shapes):
            # Limite Xavier: sqrt(6 / (fan_in + fan_out))
            limit = np.sqrt(6.0 / (cols + rows))
            size = rows * cols
            params[idx:idx + size] = np.random.uniform(
                -limit, limit, size
            ).astype(np.float32)
            idx += size
        
        # Biases inicializados com zero (padrão)
        for (rows, _) in self.biases_shapes:
            size = rows
            params[idx:idx + size] = 0.0
            idx += size
        
        return params
    
    def _he_init(self) -> np.ndarray:
        """Inicialização He - adequada para ReLU."""
        params = np.zeros(self.num_params, dtype=np.float32)
        idx = 0
        
        for i, (rows, cols) in enumerate(self.weights_shapes):
            # Limite He: sqrt(2 / fan_in)
            std = np.sqrt(2.0 / cols)
            size = rows * cols
            params[idx:idx + size] = np.random.randn(size).astype(np.float32) * std
            idx += size
        
        # Biases inicializados com zero
        for (rows, _) in self.biases_shapes:
            size = rows
            params[idx:idx + size] = 0.0
            idx += size
        
        return params

    def set_params(self, params: np.ndarray) -> None:
        if params.shape[0] != self.num_params:
            raise ValueError("Tamanho de vetor de pesos incompatível.")
        self.params = params.astype(np.float32)
        # Cacheia unpack para evitar recalcular a cada forward
        self._cached_weights = None
        self._cached_biases = None

    def get_params(self) -> np.ndarray:
        return self.params.copy()

    def _unpack_params(self):
        """Separa o vetor de params em listas de pesos e biases (com cache)."""
        # Se já está cacheado e params não mudou, reutiliza
        if self._cached_weights is not None and self._cached_biases is not None:
            return self._cached_weights, self._cached_biases
        
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
        
        # Cacheia para reutilização
        self._cached_weights = weights
        self._cached_biases = biases
        return weights, biases

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: vetor (input_dim,) - versão otimizada com cache."""
        weights, biases = self._unpack_params()
        # Usa view ao invés de reshape quando possível
        a = x.reshape(-1, 1) if x.ndim == 1 else x
        for i, (w, b) in enumerate(zip(weights, biases)):
            z = w @ a + b
            if i < len(weights) - 1:
                a = np.tanh(z)
            else:
                a = z  # camada de saída linear
        return a.flatten()
