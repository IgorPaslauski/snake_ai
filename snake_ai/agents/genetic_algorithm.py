from __future__ import annotations
from typing import List
import numpy as np
from .genome import Genome


class GeneticAlgorithm:
    """
    Algoritmo genético simples:
    - seleção por torneio
    - crossover uniforme
    - mutação gaussiana
    """

    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_std: float,
        crossover_rate: float,
        elite_fraction: float,
        rng: np.random.Generator | None = None,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.rng = rng or np.random.default_rng()

    def init_population(self, num_params: int) -> List[Genome]:
        population: List[Genome] = []
        for _ in range(self.population_size):
            params = self.rng.standard_normal(num_params).astype(np.float32) * 0.5
            population.append(Genome(params=params))
        return population

    def _tournament_select(self, population: List[Genome], k: int = 3) -> Genome:
        competitors = self.rng.choice(population, size=k, replace=False)
        best = max(competitors, key=lambda g: g.fitness if g.fitness is not None else -1e9)
        return best

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy()
        mask = self.rng.random(parent1.params.shape) < 0.5
        child_params = np.where(mask, parent1.params, parent2.params)
        return Genome(params=child_params)

    def _mutate(self, genome: Genome) -> None:
        mask = self.rng.random(genome.params.shape) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_std, size=genome.params.shape)
        genome.params = genome.params + mask * noise

    def next_generation(self, population: List[Genome]) -> List[Genome]:
        population = sorted(
            population,
            key=lambda g: g.fitness if g.fitness is not None else -1e9,
            reverse=True,
        )

        new_population: List[Genome] = []

        elite_count = max(1, int(self.elite_fraction * self.population_size))
        for i in range(elite_count):
            new_population.append(population[i].copy())

        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)

        return new_population
