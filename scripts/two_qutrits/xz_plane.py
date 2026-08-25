# ------------------------------
# Importations
# ------------------------------

# Numerical and scientific python programming
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary python functions
import time
from pathlib import Path
from typing import Any

# Local importations
from moments.bloch import (generate_gell_mann_basis, compute_tensor_basis, compute_subset_index_map,
                           compute_bipartite_region_upper, compute_bipartite_region_ent, compute_bipartite_region_lower,)
from moments.initialization import compute_initial_param_repeat
from moments.optimization import OptimizationResult, opt_moment_preserving_ent
from moments.ent_meas import compute_negativity

# Saving
from moments.saving import find_project_root

# ------------------------------
# Configurations
# ------------------------------

# Configure Matplotlib's rendering parameters.
plt.style.use('default')
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.5,
    'axes.edgecolor': 'black',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fancybox': True,
    'legend.fontsize': 20,
})

# ------------------------------
# Definitions
# ------------------------------

def basis_definitions(dim: list[int]) -> tuple[np.ndarray, dict[tuple[int, ...], np.ndarray]]:
    """
    Construct the tensor-product Pauli basis and subset index map.
    
    Parameters
    ----------
    dim : list[int]
        Dimensions of the individual quantum subsystems.
    
    Returns
    -------
    tensor_basis : numpy.ndarray
        Tensor-product basis generated from the local Pauli bases.
    subset_index_map : dict[tuple[int, ...], numpy.ndarray]
        Mapping from subsystem subsets to the corresponding Bloch-vector indices.
    """

    # Initialize the Pauli basis of a one-qubit system
    local_bases = [generate_gell_mann_basis(dn) for dn in dim]
    local_basis_sizes = [len(basis) for basis in local_bases]

    # Compute tensor basis of the N qubit system
    tensor_basis = compute_tensor_basis(local_bases)
    # Compute index mappings from basis elements to Bloch vector elements.
    subset_index_map = compute_subset_index_map(local_basis_sizes)

    return tensor_basis, subset_index_map


# ------------------------------
# Moment space discretization
# ------------------------------

def space_discretization(D: int) -> dict[str, Any]:
    """
    Create a grid for the optimizaion and parametrize the phisically allowed bloch lengths.

    Parameters
    ----------
    D : int
        The number of points used for the primary grid coordinate is ``D + 1``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the one-dimensional coordinates, mesh grids,
        physical-state indices, and masks identifying entangled, mixed,
        and separable regions.
    """
    # Determine the number of grid points along the two coordinates.
    Dx, Dz = D + 1, 2 * D + 1

    # Discretize each coordinate.
    x = np.linspace(0, np.sqrt(2), Dx)
    z = np.linspace(0, 2 * np.sqrt(2), Dz)

    # Construct the two-dimensional coordinate mesh.
    X, Z = np.meshgrid(x, z, indexing='ij')

    # Extract the indices of all physically allowed grid points for the loop.
    indices = np.ndindex(X.shape)

    return {
        "x": x,
        "z": z,
        "X": X,
        "Z": Z,
        "indices": indices
    }

# ------------------------------
# Optimization
# ------------------------------

def run_repeat(run_kwargs: dict[str, Any], attempts: int = 5) -> OptimizationResult:
    """
    Run an optimization repeatedly and retain the best result.

    The optimization direction is inferred from the ``optimization`` entry in ``run_kwargs``.
    For maximization, the result with the largest final metric is retained.
    For minimization, the result with the smallest final metric is retained.

    Parameters
    ----------
    run_kwargs : dict[str, Any]
        Keyword arguments passed to :func:`opt_moment_preserving_ent`.
    attempts : int, default=5
        Number of independent optimization attempts to perform.

    Returns
    -------
    OptimizationResult
        The optimization result with the best final metric among all attempts.

    Raises
    ------
    ValueError
        If ``attempts`` is less than one.
    """
    # Validate that at least one optimization attempt will be performed.
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    # Determine whether the optimization should maximize or minimize the metric.
    maximize = (run_kwargs["optimization"] == "maximize")

    # Initialize the best result and metric.
    best_result: OptimizationResult | None = None
    best_metric = -np.inf if maximize else np.inf

    # Repeat the optimization the requested number of times.
    for _ in range(attempts):
        # Execute one optimization run.
        result = opt_moment_preserving_ent(**run_kwargs)

        # Replace the stored result when this run improves the metric.
        if (maximize and result.metric_final > best_metric) or \
           (not maximize and result.metric_final < best_metric):
            best_metric = result.metric_final
            best_result = result

    # For static type checkers.
    assert best_result is not None
    return best_result

def main_optimization(run_kwargs: dict[str, Any], grid: dict[str, Any], data_dir: Path) -> None:
    """
    Run the entanglement optimization over every valid grid point.

    Parameters
    ----------
    run_kwargs : dict[str, Any]
        Keyword arguments passed to the optimization routine.
    grid : dict[str, Any]
        Grid coordinates and physical-region masks.
    data_dir : pathlib.Path
        Directory in which intermediate optimization results are saved.

    Returns
    -------
    ent_max : numpy.ndarray
        Array containing the optimized maximum entanglement metric.
    ent_min : numpy.ndarray
        Array containing the optimized minimum entanglement metric.
    """
    d  = int(np.prod(run_kwargs["dim"]))

    # Extract the relevant grid parameters.
    X, Z, indices = grid["X"], grid["Z"], grid["indices"]
    total_points = np.prod(X.shape)
    print(f"Number of points: {total_points}")
    
    # Initialize required storing variables.
    times: list[float] = []
    ent_max = np.full_like(X, np.nan, dtype=float)
    ent_min = np.full_like(X, np.nan, dtype=float)

    # Iterate over every physically allowed grid point.
    for counter, (idx, jdx) in enumerate(indices, 1):
        t0 = time.time()

        # Read Bloch lengths for each poitn.
        x_val = X[idx, jdx]
        z_val = Z[idx, jdx]

        # Construct Boch length constraints.
        Rt = {(1,): float(x_val), (2,): 0.0, (1, 2): float(z_val)}
        run_kwargs["Rt"] = Rt

        param_res = compute_initial_param_repeat(d, tensor_basis, subset_index_map, Rt)
        bloch_distance = sum(list(param_res.checks["moments_distance"].values()))

        if bloch_distance <= 1e-3:

            # Solve the maximization problem with a different strategy in each region.
            run_kwargs["optimization"] = "maximize"
            max_result = opt_moment_preserving_ent(**run_kwargs)
            ent_max[idx, jdx] = max_result.metric_final

            # Solve the minimization problem with a different strategy in each region.
            run_kwargs["optimization"] = "minimize"
            min_result = opt_moment_preserving_ent(**run_kwargs)
            ent_min[idx, jdx] = min_result.metric_final

        # Save the current maximum and minimum arrays as an intermediate file.
        np.savez(data_dir / "xz_ent.npz",
                 max=ent_max, min=ent_min)
        
        tf = time.time()
        times.append(tf - t0)

        percent_complete = (counter / total_points) * 100
        print(
            f"\rProgress: {percent_complete:.2f}% ({counter}/{total_points}) "
            f"| Time for this point: {tf - t0:.3f} s",
            end="",
            flush=True,
        )

    print("\n\nDone!")
    print(
        f"Total time: {sum(times) / 60:.2f} min "
        f"| Average time per point: {sum(times) / total_points:.3f} s"
    )

# ------------------------------
# Representation
# ------------------------------

def plot_results(ent_max: np.ndarray, ent_min: np.ndarray, grid: dict[str, Any], plots_dir: Path) -> None:
    """
    Plot the three-panel visualization of the entanglement bounds.

    Parameters
    ----------
    ent_max : numpy.ndarray
        Maximum optimized entanglement metric on the discretized grid.
    ent_min : numpy.ndarray
        Minimum optimized entanglement metric on the discretized grid.
    grid : dict[str, Any]
        Grid coordinates returned by :func:`space_discretization`.
    plots_dir : pathlib.Path
        Directory in which the resulting figure is saved.

    Returns
    -------
    None
        This function saves the figure to disk and does not return a value.
    """
    # Create the three-panel figure and axes.
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)

    # Extract grid coordinates.
    x, z = grid["x"], grid["z"]
    X, Z = grid["X"], grid["Z"]

    # Plot entanglement bounds.
    mesh0 = ax[0].pcolormesh(X, Z, ent_max, shading='auto', cmap='viridis', vmin=0, vmax=1)
    ax[1].pcolormesh(X, Z, ent_min, shading='auto', cmap='viridis', vmin=0, vmax=1)

    # Plot the width of the entanglement-bound interval.
    DN = np.maximum(0, ent_max - ent_min)
    mesh2 = ax[2].pcolormesh(X, Z, DN, shading='auto', cmap='viridis', vmin=0, vmax=0.42)

    # Compute the boundaries of each region.
    z_upper = compute_bipartite_region_upper(3, x, 0)
    z_ent = compute_bipartite_region_ent(3, x, 0)
    z_lower = compute_bipartite_region_lower([3, 3], x, 0)

    # Add the colorbars
    cbar0 = fig.colorbar(mesh0, ax=ax[0:2].ravel().tolist(), label=r'$\mathcal{N}$', pad=0.02, aspect=40, shrink=1)
    cbar0.set_ticks(list(np.linspace(0, 1, 6)))
    
    cbar1 = fig.colorbar(mesh2, ax=ax[2], label=r'$\Delta \mathcal{N}$', pad=0.02, aspect=40, shrink=1)
    cbar1.set_ticks(list(np.linspace(0, 0.42, 6)))

    # Configure the siplay of each panel.
    labels = ['(d)', '(e)', '(f)']
    for i in range(3):
        ax[i].set(xlabel = r"$|| r_1 || \quad (|| r_2 || = 0)$", ylabel = r"$||r_{1, 2}||$", xlim = (0, 2), ylim = (0, 3),
                  xticks=np.arange(0, 2.1, 0.5), yticks=np.arange(0, 3.1, 0.5))
        ax[i].plot(x, z_upper, 'red', linewidth=2, label='Upper bound')
        ax[i].plot(x, z_ent, 'orange', linewidth=2, label='Entangled')
        ax[i].plot(x, z_lower, 'grey', linewidth=2, label='Lower bound')
        ax[i].grid(True)
        ax[i].text(-0.25, 1.05, labels[i], transform=ax[i].transAxes, va='top', ha='left')

    # Add the legends.
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')

    # Save the final figure.
    plt.savefig(plots_dir / "xz_ent.png")

# ------------------------------
# Script entry point
# ------------------------------

if __name__ == "__main__":

    # Define paths for relevant directories.
    PROJECT_ROOT = find_project_root()
    data_dir = PROJECT_ROOT / "data" / "scripts" / "two_qutrits"
    plots_dir = PROJECT_ROOT / "plots" / "scripts" / "two_qutrits"
    # Create directories if they don't exist.
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define the dimensions of the two-qubit system.
    dn, N = 3, 2
    dim = [dn] * N
    
    # Construct the tensor basis and subset index map.
    tensor_basis, subset_index_map = basis_definitions(dim)

    # Set the grid resolution.
    D = 200
    # Construct the discretized moment-space grid.
    grid = space_discretization(D)

    # Define the fixed arguments used by the optimization routine.
    run_kwargs = {"dim": dim, "tensor_basis": tensor_basis, "subset_index_map": subset_index_map,
                  "metric": "partial_trace_norm", "cholesky_opt": True, "exact_jac": True}

    # Run the optimization over the full grid.
    main_optimization(run_kwargs, grid, data_dir)

    # Load the saved trace-norm optimization results.
    ent = np.load(data_dir / "xz_ent.npz")
    # Convert the trace-norm results into negativity values.
    neg_max = np.asarray(compute_negativity(trace_norm_pt=ent["max"]))
    neg_min = np.asarray(compute_negativity(trace_norm_pt=ent["min"]))
    # Generate and save the final visualization.
    plot_results(neg_max, neg_min, grid, plots_dir)