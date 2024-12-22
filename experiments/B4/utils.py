import json
from pathlib import Path
from typing import Dict, List

def load_json(file_path: Path) -> Dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def export_predictions(results_dir: Path, predictions: List[Dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_file: Path = results_dir / "test_predictions.jsonl"
    with open(prediction_file, "w+", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False))
            f.write("\n")