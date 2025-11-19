import argparse
from snake_ai.visualization.plots import compare_experiments


def main():
    parser = argparse.ArgumentParser(description="Comparar múltiplos experimentos.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Lista de nomes de experimentos (pastas em results/).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="fitness_max",
        choices=["fitness_mean", "fitness_max"],
        help="Métrica usada na comparação.",
    )
    args = parser.parse_args()

    compare_experiments(args.experiments, metric=args.metric)


if __name__ == "__main__":
    main()
