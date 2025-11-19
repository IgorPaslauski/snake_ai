import argparse
import numpy as np

from snake_ai.env.snake_env import SnakeConfig
from snake_ai.visualization.live_view import LiveViewer, LiveViewConfig
from snake_ai.utils.paths import get_experiment_dir


def main():
    parser = argparse.ArgumentParser(description="Reprodução do melhor agente.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Nome do experimento (pasta em results/).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Número de episódios de replay.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Multiplicador de velocidade (1.0 = normal, 2.0 = rápido).",
    )
    args = parser.parse_args()

    exp_dir = get_experiment_dir(args.experiment)
    model_path = exp_dir / "models" / "best_genome.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
    data = np.load(model_path, allow_pickle=True)

    params = data["params"]
    net_arch = data["net_arch"].tolist()

    env_cfg = SnakeConfig(
        grid_width=20,
        grid_height=20,
        max_steps_without_food=200,
        step_penalty=-0.001,
        food_reward=1.0,
        death_penalty=-1.0,
    )

    view_cfg = LiveViewConfig(cell_size=25, fps=10, show_grid=True)
    viewer = LiveViewer(env_cfg, net_arch, params, view_cfg)
    viewer.run(episodes=args.episodes, speed_multiplier=args.speed)


if __name__ == "__main__":
    main()
