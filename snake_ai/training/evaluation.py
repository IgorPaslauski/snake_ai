from __future__ import annotations
from typing import Callable
import numpy as np
from snake_ai.env.snake_env import SnakeEnv
from snake_ai.agents.neural_net import NeuralNetwork
from snake_ai.agents.genome import Genome


def evaluate_genome(
    genome: Genome,
    build_env: Callable[[], SnakeEnv],
    net_architecture: list[int],
    episodes_per_genome: int,
    max_steps_per_episode: int | None = None,
) -> float:
    """Avalia um genome em vários episódios e retorna fitness média."""
    net = NeuralNetwork(net_architecture)
    net.set_params(genome.params)

    total_fitness = 0.0
    total_apples = 0
    for _ in range(episodes_per_genome):
        env = build_env()
        state = env.reset()
        episode_fitness = 0.0
        steps = 0
        while True:
            logits = net.forward(state)
            action = int(np.argmax(logits))  # 0,1,2
            next_state, reward, done, info = env.step(action)
            episode_fitness += reward
            state = next_state
            steps += 1
            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                break
            if done:
                break
        total_fitness += episode_fitness
        total_apples += info.get("score", 0)

    fitness = total_fitness / episodes_per_genome + 0.1 * (total_apples / episodes_per_genome)
    genome.fitness = float(fitness)
    return float(fitness)
