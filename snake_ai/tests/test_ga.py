import numpy as np
from snake_ai.agents.genetic_algorithm import GeneticAlgorithm
from snake_ai.agents.genome import Genome


def test_ga_next_generation():
    ga = GeneticAlgorithm(
        population_size=10,
        mutation_rate=0.1,
        mutation_std=0.1,
        crossover_rate=0.7,
        elite_fraction=0.2,
    )
    num_params = 20
    pop = ga.init_population(num_params)
    for i, g in enumerate(pop):
        g.fitness = float(i)

    new_pop = ga.next_generation(pop)
    assert len(new_pop) == len(pop)
    for g in new_pop:
        assert g.params.shape[0] == num_params
