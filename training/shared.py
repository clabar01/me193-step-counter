"""Load the shared feature module (board/python/features.py).

It is imported by path under the name "shared_features" because a plain
`import features` from training/ would pick up training/features.py instead.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHARED_PATH = REPO / "board" / "python" / "features.py"
APP_MODELS = REPO / "board" / "python" / "models"   # what the board app loads


def shared_features():
    if "shared_features" not in sys.modules:
        spec = importlib.util.spec_from_file_location("shared_features", SHARED_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["shared_features"] = module
        spec.loader.exec_module(module)
    return sys.modules["shared_features"]


def copy_to_app(path):
    """Copy a trained model file into the board app (board/python/models/),
    so board/deploy_board.sh ships the newest models."""
    APP_MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, APP_MODELS / Path(path).name)
    return APP_MODELS / Path(path).name
