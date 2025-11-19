from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

from snake_ai.utils.paths import get_experiment_dir


def plot_training_curves(experiment_name: str) -> None:
    exp_dir = get_experiment_dir(experiment_name)
    csv_path = exp_dir / "logs" / "training_log.csv"
    if not csv_path.exists():
        print(f"Nenhum log encontrado em {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["generation"], df["fitness_mean"], label="Fitness média")
    plt.plot(df["generation"], df["fitness_max"], label="Fitness máxima")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title(f"Curvas de Fitness - {experiment_name}")
    plt.tight_layout()
    out_path = exp_dir / "plots" / "fitness.png"
    plt.savefig(out_path)
    plt.close()

    plt.figure()
    plt.plot(df["generation"], df["apples_mean"], label="Maçãs médias")
    plt.plot(df["generation"], df["apples_max"], label="Maçãs máximas")
    plt.xlabel("Geração")
    plt.ylabel("Maçãs (aprox.)")
    plt.legend()
    plt.title(f"Tamanho da Cobra / Maçãs - {experiment_name}")
    plt.tight_layout()
    out_path = exp_dir / "plots" / "apples.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Gráficos salvos em {exp_dir / 'plots'}")


def compare_experiments(experiment_names: List[str], metric: str = "fitness_max") -> None:
    plt.figure()
    for name in experiment_names:
        exp_dir = get_experiment_dir(name)
        csv_path = exp_dir / "logs" / "training_log.csv"
        if not csv_path.exists():
            print(f"[WARN] Sem log para experimento {name}, pulando.")
            continue
        df = pd.read_csv(csv_path)
        plt.plot(df["generation"], df[metric], label=name)

    plt.xlabel("Geração")
    plt.ylabel(metric)
    plt.legend()
    plt.title(f"Comparação de experimentos ({metric})")
    plt.tight_layout()

    out_dir = Path("results") / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_{metric}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Gráfico de comparação salvo em {out_path}")
