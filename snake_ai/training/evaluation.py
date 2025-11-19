from __future__ import annotations
from typing import Callable
import os
import numpy as np

# Suprime mensagens do pygame (deve ser antes de qualquer import do pygame)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

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
    """Avalia um genome em vários episódios e retorna fitness média (otimizado)."""
    # Cria a rede neural uma vez e reutiliza
    net = NeuralNetwork(net_architecture)
    net.set_params(genome.params)  # Isso cacheia unpack_params

    total_fitness = 0.0
    total_apples = 0
    total_steps = 0
    
    # Pré-aloca variáveis para evitar alocações no loop
    episodes = episodes_per_genome
    
    for _ in range(episodes):
        env = build_env()
        state = env.reset()
        episode_fitness = 0.0
        steps = 0
        max_steps = max_steps_per_episode if max_steps_per_episode is not None else float('inf')
        
        while steps < max_steps:
            # Forward otimizado com cache
            logits = net.forward(state)
            action = int(np.argmax(logits))  # 0,1,2
            next_state, reward, done, info = env.step(action)
            episode_fitness += reward
            state = next_state
            steps += 1
            if done:
                break
        
        total_fitness += episode_fitness
        total_apples += info.get("score", 0)
        total_steps += steps

    # Fitness melhorada: considera maçãs, passos e eficiência
    avg_fitness = total_fitness / episodes
    avg_apples = total_apples / episodes
    avg_steps = total_steps / episodes
    
    # Bônus por eficiência (mais maçãs com menos passos) - otimizado
    efficiency_bonus = 0.0
    if avg_steps > 0:
        efficiency_bonus = 0.05 * (avg_apples / avg_steps)
    
    fitness = avg_fitness + 0.1 * avg_apples + efficiency_bonus
    genome.fitness = float(fitness)
    return float(fitness)


def _evaluate_genome_worker(args):
    """Função auxiliar para avaliação paralela."""
    # Garante que a variável de ambiente está configurada nos workers
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    genome_params, env_config_dict, net_architecture, episodes_per_genome, max_steps_per_episode = args
    
    from snake_ai.env.snake_env import SnakeConfig, SnakeEnv
    
    # Reconstrói o ambiente a partir do dicionário de configuração
    # SnakeConfig é um dataclass, então podemos reconstruí-lo diretamente
    env_config = SnakeConfig(
        grid_width=env_config_dict["grid_width"],
        grid_height=env_config_dict["grid_height"],
        max_steps_without_food=env_config_dict["max_steps_without_food"],
        step_penalty=env_config_dict["step_penalty"],
        food_reward=env_config_dict["food_reward"],
        death_penalty=env_config_dict["death_penalty"],
    )
    
    def build_env():
        return SnakeEnv(env_config)
    
    # Cria um genome temporário para avaliação
    temp_genome = Genome(params=genome_params)
    
    # Avalia o genome
    fitness = evaluate_genome(
        temp_genome,
        build_env=build_env,
        net_architecture=net_architecture,
        episodes_per_genome=episodes_per_genome,
        max_steps_per_episode=max_steps_per_episode,
    )
    
    return fitness
