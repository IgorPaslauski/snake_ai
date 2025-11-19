from __future__ import annotations
from typing import List
import numpy as np
from .genome import Genome


class GeneticAlgorithm:
    """
    Algoritmo genético melhorado:
    - seleção por torneio
    - crossover uniforme
    - mutação gaussiana adaptativa
    - taxa de mutação adaptativa baseada no progresso
    """

    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_std: float,
        crossover_rate: float,
        elite_fraction: float,
        rng: np.random.Generator | None = None,
        adaptive_mutation: bool = True,
    ):
        self.population_size = population_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.base_mutation_std = mutation_std
        self.mutation_std = mutation_std
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.rng = rng or np.random.default_rng()
        self.adaptive_mutation = adaptive_mutation
        self.generations_without_improvement = 0
        self.best_fitness_history = []

    def init_population(self, num_params: int, init_method: str = "xavier") -> List[Genome]:
        """
        Inicializa população com pesos aleatórios.
        init_method: "xavier" para inicialização Xavier, "normal" para normal padrão
        """
        population: List[Genome] = []
        for _ in range(self.population_size):
            if init_method == "xavier":
                # Inicialização Xavier: valores entre -sqrt(6/n) e sqrt(6/n)
                # Para simplificar, usamos uma aproximação baseada no número de parâmetros
                limit = np.sqrt(6.0 / num_params)
                params = self.rng.uniform(-limit, limit, num_params).astype(np.float32)
            else:
                # Inicialização normal padrão (compatibilidade)
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
    
    def update_adaptive_mutation(self, best_fitness: float) -> None:
        """Atualiza taxa de mutação adaptativamente baseada no progresso."""
        if not self.adaptive_mutation:
            return
        
        self.best_fitness_history.append(best_fitness)
        
        # Se não houve melhoria nas últimas 10 gerações, aumenta exploração
        if len(self.best_fitness_history) > 10:
            recent_best = max(self.best_fitness_history[-10:])
            if recent_best <= max(self.best_fitness_history[:-10] or [recent_best]):
                self.generations_without_improvement += 1
                # Aumenta mutação gradualmente
                self.mutation_rate = min(
                    self.base_mutation_rate * 1.5,
                    self.mutation_rate * 1.05
                )
                self.mutation_std = min(
                    self.base_mutation_std * 1.2,
                    self.mutation_std * 1.02
                )
            else:
                # Reduz mutação quando há progresso
                self.generations_without_improvement = 0
                self.mutation_rate = max(
                    self.base_mutation_rate * 0.5,
                    self.mutation_rate * 0.98
                )
                self.mutation_std = max(
                    self.base_mutation_std * 0.8,
                    self.mutation_std * 0.99
                )

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
