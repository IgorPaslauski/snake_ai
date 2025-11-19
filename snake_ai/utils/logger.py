import csv
from pathlib import Path
from typing import Dict, List, Any


class CsvLogger:
    """
    Logger simples para salvar métricas por geração em CSV.
    """

    def __init__(self, csv_path: Path, fieldnames: List[str]):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        if not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        # cria cabeçalho
        if not csv_path.exists():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log_row(self, row: Dict[str, Any]) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
