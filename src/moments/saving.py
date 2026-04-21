from pathlib import Path

def find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found).")

PROJECT_ROOT = find_project_root()

DATA_RAW  = PROJECT_ROOT / "data" / "raw"
DATA_PLOTS = PROJECT_ROOT / "data" / "plots"
NOTEBOOKS  = PROJECT_ROOT / "notebooks"

def get_experiment_paths(calling_notebook: str) -> tuple[Path, Path]:
    """
    Given the path of the calling notebook, returns:
      - data_dir  : data/raw/path_to_experiment/notebook_name/
      - plots_dir : data/plots/path_to_experiment/notebook_name/

    Works for both notebooks/experiments/... and notebooks/post/...
    since both mirror the same data/raw/ structure.

    Usage in a .py file:
        data_dir, plots_dir = get_experiment_paths(__file__)

    Usage in a Jupyter notebook:
        data_dir, plots_dir = get_experiment_paths(__vsc_ipynb_file__)
    """
    notebook_path = Path(calling_notebook).resolve()
    

    # Strip everything up to and including "experiments/" or "post/"
    parts = notebook_path.parts
    for anchor in ("experiments", "post"):
        if anchor in parts:
            idx = parts.index(anchor)
            # path_to_experiment/notebook_name (no suffix)
            relative = Path(*parts[idx + 1:]).with_suffix("")
            break
    else:
        raise ValueError(
            f"Notebook is not under 'experiments/' or 'post/': {notebook_path}"
        )

    data_dir  = DATA_RAW   / relative
    plots_dir = DATA_PLOTS / relative

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, plots_dir