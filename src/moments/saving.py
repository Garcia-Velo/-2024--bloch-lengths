#------------------------------
# Importations
#------------------------------

# Auxiliary python functions
from pathlib import Path

def find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found).")