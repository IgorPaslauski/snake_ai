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
        fitness_sharing: bool = True,
        sharing_sigma: float = 0.1,
        hall_of_fame_size: int = 10,
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
        
        # Fitness Sharing: mantém diversidade penalizando indivíduos similares
        self.fitness_sharing = fitness_sharing
        self.sharing_sigma = sharing_sigma  # Distância de compartilhamento
        
        # Hall of Fame: mantém melhores indivíduos históricos
        self.hall_of_fame_size = hall_of_fame_size
        self.hall_of_fame: List[Genome] = []

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

    def _distance(self, genome1: Genome, genome2: Genome) -> float:
        """Calcula distância euclidiana normalizada entre dois genomas."""
        diff = genome1.params - genome2.params
        return float(np.sqrt(np.sum(diff ** 2)) / len(diff))
    
    def _apply_fitness_sharing(self, population: List[Genome]) -> None:
        """Aplica fitness sharing para manter diversidade."""
        if not self.fitness_sharing:
            # Se não usar sharing, shared_fitness = fitness original
            for genome in population:
                genome.shared_fitness = genome.fitness
            return
        
        # Calcula fitness compartilhado para cada indivíduo
        for i, genome in enumerate(population):
            if genome.fitness is None:
                genome.shared_fitness = None
                continue
            
            sharing_sum = 1.0  # Começa com 1 (próprio indivíduo)
            
            # Soma contribuições de indivíduos similares
            for j, other in enumerate(population):
                if i != j and other.fitness is not None:
                    dist = self._distance(genome, other)
                    if dist < self.sharing_sigma:
                        # Função de sharing: 1 - (dist/sigma)^2
                        sharing_value = 1.0 - (dist / self.sharing_sigma) ** 2
                        sharing_sum += max(0.0, sharing_value)
            
            # Fitness compartilhado = fitness original / soma de sharing
            genome.shared_fitness = genome.fitness / sharing_sum
    
    def _update_hall_of_fame(self, population: List[Genome]) -> None:
        """Atualiza Hall of Fame com os melhores indivíduos históricos (usa fitness original)."""
        # Adiciona melhores da população atual ao Hall of Fame
        sorted_pop = sorted(
            population,
            key=lambda g: g.fitness if g.fitness is not None else -1e9,
            reverse=True
        )
        
        for genome in sorted_pop[:5]:  # Top 5 da geração atual
            if genome.fitness is None:
                continue
            
            # Adiciona se não está no Hall of Fame ou é melhor
            is_duplicate = False
            for hof_genome in self.hall_of_fame:
                if self._distance(genome, hof_genome) < 0.01:  # Muito similar
                    if genome.fitness > (hof_genome.fitness or -1e9):
                        # Substitui se é melhor
                        self.hall_of_fame.remove(hof_genome)
                        self.hall_of_fame.append(genome.copy())
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.hall_of_fame.append(genome.copy())
        
        # Mantém apenas os melhores no Hall of Fame (baseado em fitness original)
        self.hall_of_fame = sorted(
            self.hall_of_fame,
            key=lambda g: g.fitness if g.fitness is not None else -1e9,
            reverse=True
        )[:self.hall_of_fame_size]
    
    def _tournament_select(self, population: List[Genome], k: int = 3) -> Genome:
        competitors = self.rng.choice(population, size=k, replace=False)
        # Usa shared_fitness para seleção (se disponível), senão usa fitness original
        best = max(competitors, key=lambda g: (
            g.shared_fitness if g.shared_fitness is not None 
            else (g.fitness if g.fitness is not None else -1e9)
        ))
        return best

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy()
        
        # Crossover aritmético (blend) - cria filhos mais diversos
        # Combina os pais de forma suave ao invés de apenas trocar genes
        alpha = self.rng.random()  # Fator de blend aleatório
        child_params = alpha * parent1.params + (1 - alpha) * parent2.params
        
        # Adiciona pequena variação para aumentar diversidade
        noise = self.rng.normal(0.0, 0.01, size=parent1.params.shape)
        child_params = child_params + noise
        
        return Genome(params=child_params.astype(np.float32))

    def _mutate(self, genome: Genome) -> None:
        mask = self.rng.random(genome.params.shape) < self.mutation_rate
        # Mutação gaussiana com chance de mutação mais forte (10% das mutações)
        strong_mutation_mask = self.rng.random(genome.params.shape) < 0.1
        noise = self.rng.normal(0.0, self.mutation_std, size=genome.params.shape)
        # Mutações fortes têm 3x o desvio padrão
        strong_noise = self.rng.normal(0.0, self.mutation_std * 3.0, size=genome.params.shape)
        final_noise = np.where(strong_mutation_mask, strong_noise, noise)
        genome.params = genome.params + mask * final_noise
    
    def update_adaptive_mutation(self, best_fitness: float) -> None:
        """Atualiza taxa de mutação adaptativamente baseada no progresso."""
        if not self.adaptive_mutation:
            return
        
        self.best_fitness_history.append(best_fitness)
        
        # Se não houve melhoria nas últimas 20 gerações, aumenta exploração mais agressivamente
        if len(self.best_fitness_history) > 20:
            recent_best = max(self.best_fitness_history[-20:])
            previous_best = max(self.best_fitness_history[:-20] or [recent_best])
            
            # Verifica se realmente estagnou (com tolerância pequena)
            if recent_best <= previous_best + 0.1:  # Tolerância de 0.1
                self.generations_without_improvement += 1
                # Aumenta mutação mais agressivamente quando estagnado
                self.mutation_rate = min(
                    self.base_mutation_rate * 2.0,  # Pode dobrar
                    self.mutation_rate * 1.1  # Aumenta 10% por geração
                )
                self.mutation_std = min(
                    self.base_mutation_std * 2.0,
                    self.mutation_std * 1.05
                )
            else:
                # Reduz mutação mais lentamente quando há progresso
                self.generations_without_improvement = 0
                self.mutation_rate = max(
                    self.base_mutation_rate * 0.7,  # Não reduz tanto
                    self.mutation_rate * 0.995  # Reduz muito lentamente
                )
                self.mutation_std = max(
                    self.base_mutation_std * 0.9,
                    self.mutation_std * 0.998
                )

    def next_generation(self, population: List[Genome]) -> List[Genome]:
        # Aplica fitness sharing (calcula shared_fitness, mantém fitness original)
        self._apply_fitness_sharing(population)
        
        # Atualiza Hall of Fame com fitness original (não compartilhado)
        self._update_hall_of_fame(population)
        
        # Ordena por fitness compartilhado (para seleção)
        population = sorted(
            population,
            key=lambda g: (
                g.shared_fitness if g.shared_fitness is not None 
                else (g.fitness if g.fitness is not None else -1e9)
            ),
            reverse=True,
        )

        new_population: List[Genome] = []

        # Reduz elite para manter mais diversidade
        elite_count = max(1, int(self.elite_fraction * self.population_size))
        for i in range(elite_count):
            new_population.append(population[i].copy())
        
        # Adiciona alguns indivíduos do Hall of Fame (10% da população)
        hof_count = max(1, int(0.1 * self.population_size))
        if self.hall_of_fame:
            for i in range(min(hof_count, len(self.hall_of_fame))):
                hof_genome = self.hall_of_fame[i % len(self.hall_of_fame)]
                new_population.append(hof_genome.copy())
        
        # Adiciona alguns indivíduos aleatórios da população (diversidade)
        diversity_count = max(1, int(0.05 * self.population_size))  # 5% aleatórios
        for _ in range(diversity_count):
            random_idx = self.rng.integers(0, len(population))
            new_population.append(population[random_idx].copy())

        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population, k=5)  # Torneio maior = mais diversidade
            parent2 = self._tournament_select(population, k=5)
            # Evita crossover entre indivíduos muito similares
            attempts = 0
            while (parent1 is parent2 or 
                   self._distance(parent1, parent2) < 0.05) and attempts < 5:
                parent2 = self._tournament_select(population, k=5)
                attempts += 1
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)

        return new_population
