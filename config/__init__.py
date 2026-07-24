import os
from pathlib import Path

def check_paths(*paths) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
