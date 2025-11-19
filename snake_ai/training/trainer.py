from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Suprime mensagens do pygame (deve ser antes de qualquer import do pygame)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from snake_ai.env.snake_env import SnakeConfig, SnakeEnv
from snake_ai.agents.neural_net import NeuralNetwork
from snake_ai.agents.genetic_algorithm import GeneticAlgorithm
from snake_ai.agents.genome import Genome
from snake_ai.training.evaluation import evaluate_genome, _evaluate_genome_worker
from snake_ai.utils.paths import get_experiment_dir
from snake_ai.utils.logger import CsvLogger
from snake_ai.visualization.live_view import LiveViewer, LiveViewConfig
from snake_ai.visualization.plots import plot_training_curves


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
    self_body_penalty: float

    show_live: bool
    live_fps: int
    live_speed: float
    live_cell_size: int
    live_episodes_per_gen: int
    live_every_n_generations: int

    save_logs: bool
    save_best_model: bool
    
    # Novas opções de otimização
    use_parallel: bool = True
    num_workers: int | None = None  # None = usar todos os cores disponíveis
    early_stopping_patience: int | None = None  # None = desabilitado
    early_stopping_min_delta: float = 0.001  # Melhoria mínima para resetar contador
    checkpoint_every_n_generations: int | None = None  # None = desabilitado
    adaptive_mutation: bool = True
    plot_every_n_generations: int | None = 5  # Plotar gráficos a cada N gerações (None = só no final)


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
            self_body_penalty=cfg.self_body_penalty,
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
            adaptive_mutation=cfg.adaptive_mutation,
            fitness_sharing=True,  # Ativa fitness sharing para manter diversidade
            sharing_sigma=0.15,  # Distância de compartilhamento
            hall_of_fame_size=15,  # Mantém 15 melhores históricos
        )

        # Usa inicialização Xavier para melhor convergência
        tmp_net = NeuralNetwork(self.net_arch, init_method="xavier")
        self.population: List[Genome] = self.ga.init_population(tmp_net.num_params, init_method="xavier")
        
        # Configuração de paralelização
        self.use_parallel = cfg.use_parallel
        self.num_workers = cfg.num_workers if cfg.num_workers is not None else cpu_count()
        
        # Pool de workers reutilizável (criado na primeira avaliação)
        self._worker_pool = None
        
        # Early stopping
        self.early_stopping_patience = cfg.early_stopping_patience
        self.early_stopping_min_delta = cfg.early_stopping_min_delta
        self.best_fitness_so_far = float('-inf')
        self.generations_without_improvement = 0
        
        # Checkpointing
        self.checkpoint_every_n = cfg.checkpoint_every_n_generations
        
        # Plotagem periódica
        self.plot_every_n = cfg.plot_every_n_generations

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
    
    def _evaluate_population_parallel(self, population: List[Genome]) -> List[float]:
        """Avalia população em paralelo usando multiprocessing (pool reutilizável)."""
        # Prepara argumentos para workers
        # Converte SnakeConfig para dicionário manualmente para garantir serialização
        env_config_dict = {
            "grid_width": self.env_config.grid_width,
            "grid_height": self.env_config.grid_height,
            "max_steps_without_food": self.env_config.max_steps_without_food,
            "step_penalty": self.env_config.step_penalty,
            "food_reward": self.env_config.food_reward,
            "death_penalty": self.env_config.death_penalty,
            "self_body_penalty": self.env_config.self_body_penalty,
        }
        args_list = [
            (
                genome.params.copy(),
                env_config_dict,
                self.net_arch,
                self.cfg.episodes_per_genome,
                self.cfg.max_steps_per_episode,
            )
            for genome in population
        ]
        
        # Reutiliza pool de workers para evitar overhead de criação/destruição
        if self._worker_pool is None:
            self._worker_pool = Pool(processes=self.num_workers)
        
        # Avalia em paralelo
        fitnesses = self._worker_pool.map(_evaluate_genome_worker, args_list)
        
        # Atualiza fitness dos genomas
        for genome, fitness in zip(population, fitnesses):
            genome.fitness = fitness
        
        return fitnesses
    
    def __del__(self):
        """Fecha pool de workers ao destruir trainer."""
        if hasattr(self, '_worker_pool') and self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool.join()
    
    def _evaluate_population_sequential(self, population: List[Genome]) -> List[float]:
        """Avalia população sequencialmente (fallback)."""
        fitnesses = []
        for genome in tqdm(population, desc="Avaliando genomas", leave=False):
            fitness = evaluate_genome(
                genome,
                build_env=self.build_env,
                net_architecture=self.net_arch,
                episodes_per_genome=self.cfg.episodes_per_genome,
                max_steps_per_episode=self.cfg.max_steps_per_episode,
            )
            fitnesses.append(fitness)
        return fitnesses
    
    def _save_checkpoint(self, generation: int, population: List[Genome], best_genome: Genome) -> None:
        """Salva checkpoint do treinamento."""
        checkpoint_dir = self.exp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_gen_{generation}.npz"
        
        # Salva população e melhor genoma
        population_params = [g.params for g in population]
        population_fitnesses = [g.fitness if g.fitness is not None else -1e9 for g in population]
        
        np.savez(
            checkpoint_path,
            generation=generation,
            population_params=population_params,
            population_fitnesses=population_fitnesses,
            best_params=best_genome.params,
            best_fitness=best_genome.fitness if best_genome.fitness is not None else -1e9,
            net_arch=np.array(self.net_arch),
            config=json.dumps(self.cfg.__dict__),
        )
    
    def _load_checkpoint(self, checkpoint_path: Path) -> tuple[int, List[Genome], Genome]:
        """Carrega checkpoint do treinamento."""
        data = np.load(checkpoint_path, allow_pickle=True)
        
        generation = int(data['generation'])
        population_params = data['population_params']
        population_fitnesses = data['population_fitnesses']
        
        # Reconstrói população
        population = [
            Genome(params=params, fitness=float(fitness))
            for params, fitness in zip(population_params, population_fitnesses)
        ]
        
        # Reconstrói melhor genoma
        best_genome = Genome(
            params=data['best_params'],
            fitness=float(data['best_fitness'])
        )
        
        return generation, population, best_genome

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
        
        # Barra de progresso principal
        pbar = tqdm(
            range(self.cfg.generations),
            desc=f"Treinando {self.cfg.experiment_name}",
            unit="geração"
        )

        for gen in pbar:
            # Avalia população (paralelo ou sequencial)
            if self.use_parallel:
                fitnesses = self._evaluate_population_parallel(self.population)
            else:
                fitnesses = self._evaluate_population_sequential(self.population)
            
            apples_list = [max(0.0, (f / self.env_config.food_reward)) for f in fitnesses]

            fitness_mean = float(np.mean(fitnesses))
            fitness_max = float(np.max(fitnesses))
            apples_mean = float(np.mean(apples_list))
            apples_max = float(np.max(apples_list))

            # Atualiza barra de progresso
            pbar.set_postfix({
                'fitness_média': f'{fitness_mean:.3f}',
                'fitness_máx': f'{fitness_max:.3f}',
                'maçãs_máx': f'{apples_max:.1f}',
            })

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
            gen_best_fitness = gen_best.fitness or -1e9
            
            # Atualiza melhor overall
            if best_overall is None or gen_best_fitness > (best_overall.fitness or -1e9):
                best_overall = gen_best.copy()
                if self.cfg.save_best_model:
                    self._save_best_model(best_overall, gen)
            
            # Early stopping
            if self.early_stopping_patience is not None:
                improvement = gen_best_fitness - self.best_fitness_so_far
                if improvement > self.early_stopping_min_delta:
                    self.best_fitness_so_far = gen_best_fitness
                    self.generations_without_improvement = 0
                else:
                    self.generations_without_improvement += 1
                
                if self.generations_without_improvement >= self.early_stopping_patience:
                    print(
                        f"\nEarly stopping ativado após {gen+1} gerações. "
                        f"Sem melhoria por {self.early_stopping_patience} gerações."
                    )
                    break
            
            # Atualiza mutação adaptativa
            if self.cfg.adaptive_mutation:
                self.ga.update_adaptive_mutation(gen_best_fitness)
                if gen % 10 == 0:  # Mostra taxa de mutação a cada 10 gerações
                    pbar.write(
                        f"Taxa de mutação adaptativa: {self.ga.mutation_rate:.4f}, "
                        f"std: {self.ga.mutation_std:.4f}"
                    )

            if (
                self.cfg.show_live
                and self.cfg.live_every_n_generations > 0
                and gen % self.cfg.live_every_n_generations == 0
            ):
                self._show_generation_best(gen_best, gen)

            # Checkpointing
            if self.checkpoint_every_n is not None and gen > 0 and gen % self.checkpoint_every_n == 0:
                self._save_checkpoint(gen, self.population, best_overall)
                pbar.write(f"Checkpoint salvo na geração {gen+1}")

            # Plotagem periódica de gráficos (menos frequente para não impactar performance)
            if self.plot_every_n is not None and self.cfg.save_logs:
                if gen > 0 and (gen % self.plot_every_n == 0 or gen == self.cfg.generations - 1):
                    try:
                        # Plotagem em background para não bloquear
                        plot_training_curves(self.cfg.experiment_name)
                        if gen % (self.plot_every_n * 2) == 0:  # Só mostra mensagem a cada 2 plots
                            pbar.write(f"Gráficos atualizados na geração {gen+1}")
                    except Exception as e:
                        pbar.write(f"Erro ao plotar gráficos: {e}")

            self.population = self.ga.next_generation(self.population)

        pbar.close()
        
        # Fecha pool de workers explicitamente
        if self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool.join()
            self._worker_pool = None
        
        assert best_overall is not None
        return best_overall
