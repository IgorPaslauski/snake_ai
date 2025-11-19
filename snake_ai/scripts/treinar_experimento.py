import argparse
from datetime import datetime
from snake_ai.utils.config_loader import load_config
from snake_ai.training.trainer import Trainer, TrainingConfig
from snake_ai.visualization.plots import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Treinar experimento Snake AI.")
    parser.add_argument(
        "--config",
        type=str,
        default="snake_ai/config/experimento_default.yaml",
        help="Caminho para arquivo de configuração (.yaml/.json).",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Não adicionar timestamp ao nome do experimento (usa nome original).",
    )
    args = parser.parse_args()

    cfg_dict = load_config(args.config)

    visual_cfg = cfg_dict.get("visual", {})
    logging_cfg = cfg_dict.get("logging", {})
    optimization_cfg = cfg_dict.get("optimization", {})

    # Adiciona timestamp ao nome do experimento para criar pasta única
    base_name = cfg_dict["experiment"]["name"]
    if not args.no_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{base_name}_{timestamp}"
    else:
        experiment_name = base_name

    tcfg = TrainingConfig(
        experiment_name=experiment_name,
        random_seed=cfg_dict["experiment"]["random_seed"],
        population_size=cfg_dict["ga"]["population_size"],
        generations=cfg_dict["ga"]["generations"],
        episodes_per_genome=cfg_dict["ga"]["episodes_per_genome"],
        mutation_rate=cfg_dict["ga"]["mutation_rate"],
        mutation_std=cfg_dict["ga"]["mutation_std"],
        crossover_rate=cfg_dict["ga"]["crossover_rate"],
        elite_fraction=cfg_dict["ga"]["elite_fraction"],
        grid_width=cfg_dict["env"]["grid_width"],
        grid_height=cfg_dict["env"]["grid_height"],
        max_steps_per_episode=cfg_dict["env"]["max_steps_per_episode"],
        max_steps_without_food=cfg_dict["env"]["max_steps_without_food"],
        step_penalty=cfg_dict["env"]["step_penalty"],
        food_reward=cfg_dict["env"]["food_reward"],
        death_penalty=cfg_dict["env"]["death_penalty"],
        self_body_penalty=cfg_dict["env"].get("self_body_penalty", -0.5),
        show_live=visual_cfg.get("show_live", False),
        live_fps=visual_cfg.get("live_fps", 10),
        live_speed=visual_cfg.get("live_speed", 1.0),
        live_cell_size=visual_cfg.get("live_cell_size", 20),
        live_episodes_per_gen=visual_cfg.get("live_episodes_per_gen", 1),
        live_every_n_generations=visual_cfg.get("live_every_n_generations", 1),
        save_logs=logging_cfg.get("save_logs", True),
        save_best_model=logging_cfg.get("save_best_model", True),
        # Novas opções de otimização
        use_parallel=optimization_cfg.get("use_parallel", True),
        num_workers=optimization_cfg.get("num_workers", None),
        early_stopping_patience=optimization_cfg.get("early_stopping_patience", None),
        early_stopping_min_delta=optimization_cfg.get("early_stopping_min_delta", 0.001),
        checkpoint_every_n_generations=optimization_cfg.get("checkpoint_every_n_generations", None),
        adaptive_mutation=optimization_cfg.get("adaptive_mutation", True),
        plot_every_n_generations=optimization_cfg.get("plot_every_n_generations", 5),
    )

    print(f"Iniciando treinamento: {experiment_name}")
    print(f"Resultados serão salvos em: results/{experiment_name}/")

    trainer = Trainer(tcfg)
    trainer.run()

    # Plota gráficos finais (caso não tenha sido plotado recentemente)
    if tcfg.save_logs:
        plot_training_curves(tcfg.experiment_name)
        print(f"\nTreinamento concluído! Resultados em: results/{experiment_name}/")


if __name__ == "__main__":
    main()
