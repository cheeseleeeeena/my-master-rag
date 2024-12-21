import json
from pathlib import Path
from typing import Dict

def load_json(file_path: Path) -> Dict:
    """Utility to load JSON files."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)