from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json
import numpy as np

from snake_ai.env.snake_env import SnakeConfig, SnakeEnv
from snake_ai.agents.neural_net import NeuralNetwork
from snake_ai.agents.genetic_algorithm import GeneticAlgorithm
from snake_ai.agents.genome import Genome
from snake_ai.training.evaluation import evaluate_genome
from snake_ai.utils.paths import get_experiment_dir
from snake_ai.utils.logger import CsvLogger
from snake_ai.visualization.live_view import LiveViewer, LiveViewConfig


@dataclass
class TrainingConfig:
    experiment_name: str
    random_seed: int
    population_size: int
    generations: int
    episodes_per_genome: int
    mutation_rate: float
    mutation_std: float
    crossover_rate: float
    elite_fraction: float
    grid_width: int
    grid_height: int
    max_steps_per_episode: int | None
    max_steps_without_food: int
    step_penalty: float
    food_reward: float
    death_penalty: float

    show_live: bool
    live_fps: int
    live_speed: float
    live_cell_size: int
    live_episodes_per_gen: int
    live_every_n_generations: int

    save_logs: bool
    save_best_model: bool


class Trainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.exp_dir = get_experiment_dir(cfg.experiment_name)

        self.env_config = SnakeConfig(
            grid_width=cfg.grid_width,
            grid_height=cfg.grid_height,
            max_steps_without_food=cfg.max_steps_without_food,
            step_penalty=cfg.step_penalty,
            food_reward=cfg.food_reward,
            death_penalty=cfg.death_penalty,
        )

        self.input_dim = 11
        self.hidden_layers = [16, 16]
        self.output_dim = 3
        self.net_arch = [self.input_dim] + self.hidden_layers + [self.output_dim]

        self.ga = GeneticAlgorithm(
            population_size=cfg.population_size,
            mutation_rate=cfg.mutation_rate,
            mutation_std=cfg.mutation_std,
            crossover_rate=cfg.crossover_rate,
            elite_fraction=cfg.elite_fraction,
            rng=self.rng,
        )

        tmp_net = NeuralNetwork(self.net_arch)
        self.population: List[Genome] = self.ga.init_population(tmp_net.num_params)

        if cfg.save_logs:
            self.logger = CsvLogger(
                self.exp_dir / "logs" / "training_log.csv",
                fieldnames=[
                    "generation",
                    "fitness_mean",
                    "fitness_max",
                    "apples_mean",
                    "apples_max",
                ],
            )
        else:
            self.logger = None

        with open(self.exp_dir / "config_used.json", "w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, indent=2)

    def build_env(self) -> SnakeEnv:
        return SnakeEnv(self.env_config)

    def _save_best_model(self, genome: Genome, generation: int) -> None:
        model_path = self.exp_dir / "models" / "best_genome.npz"
        np.savez(
            model_path,
            params=genome.params,
            fitness=genome.fitness if genome.fitness is not None else 0.0,
            generation=generation,
            net_arch=np.array(self.net_arch),
        )
        print(f"Melhor modelo salvo em: {model_path}")

    def _show_generation_best(self, genome: Genome, generation_index: int) -> None:
        print(
            f"[{self.cfg.experiment_name}] Exibindo melhor da geração {generation_index + 1} "
            f"(fitness={genome.fitness:.3f} aprox.)"
        )
        view_cfg = LiveViewConfig(
            cell_size=self.cfg.live_cell_size,
            fps=self.cfg.live_fps,
            show_grid=True,
        )
        viewer = LiveViewer(self.env_config, self.net_arch, genome.params, view_cfg)
        viewer.run(
            episodes=self.cfg.live_episodes_per_gen,
            speed_multiplier=self.cfg.live_speed,
        )

    def run(self) -> Genome:
        best_overall: Genome | None = None

        for gen in range(self.cfg.generations):
            fitnesses = []
            apples_list = []

            for genome in self.population:
                fitness = evaluate_genome(
                    genome,
                    build_env=self.build_env,
                    net_architecture=self.net_arch,
                    episodes_per_genome=self.cfg.episodes_per_genome,
                    max_steps_per_episode=self.cfg.max_steps_per_episode,
                )
                fitnesses.append(fitness)
                apples_list.append(max(0.0, (fitness / self.env_config.food_reward)))

            fitness_mean = float(np.mean(fitnesses))
            fitness_max = float(np.max(fitnesses))
            apples_mean = float(np.mean(apples_list))
            apples_max = float(np.max(apples_list))

            print(
                f"[{self.cfg.experiment_name}] Geração {gen+1}/{self.cfg.generations} "
                f"- fitness: média={fitness_mean:.3f}, máx={fitness_max:.3f}"
            )

            if self.logger:
                self.logger.log_row(
                    {
                        "generation": gen,
                        "fitness_mean": fitness_mean,
                        "fitness_max": fitness_max,
                        "apples_mean": apples_mean,
                        "apples_max": apples_max,
                    }
                )

            gen_best = max(self.population, key=lambda g: g.fitness or -1e9)
            if best_overall is None or (gen_best.fitness or -1e9) > (best_overall.fitness or -1e9):
                best_overall = gen_best.copy()
                if self.cfg.save_best_model:
                    self._save_best_model(best_overall, gen)

            if (
                self.cfg.show_live
                and self.cfg.live_every_n_generations > 0
                and gen % self.cfg.live_every_n_generations == 0
            ):
                self._show_generation_best(gen_best, gen)

            self.population = self.ga.next_generation(self.population)

        assert best_overall is not None
        return best_overall
