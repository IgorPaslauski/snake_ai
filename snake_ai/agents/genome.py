from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Genome:
    """Representa um indivíduo da população (vetor de pesos da rede)."""
    params: np.ndarray
    fitness: float | None = None
    shared_fitness: float | None = None  # Fitness após fitness sharing (para seleção)

    def copy(self) -> "Genome":
        return Genome(
            params=self.params.copy(), 
            fitness=self.fitness,
            shared_fitness=self.shared_fitness
        )
