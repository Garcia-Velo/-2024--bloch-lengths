#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np
from scipy.linalg import cholesky as compute_cholesky
from scipy.optimize import minimize

# Auxiliary python functions
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Any

# Local importations
from moments.bloch import compute_bloch_vector, compute_bloch_norms_from_vector
from moments.initialization import ParameterResult, compute_initial_param_repeat, compute_param_from_X, compute_X_from_param, compute_dm_from_X
from moments.quantum import compute_is_valid_dm, compute_concurrence, compute_eof, compute_partial_trace_norm, compute_negativity

# ----------------------------------------
# Definitions
# ----------------------------------------

@dataclass
class OptimizationResult:
    """
    Data class to store the results of the moment-preserving entanglement optimization.

    Attributes
    ----------
    rho_initial : np.ndarray
        The initial density matrix.
    rho_final : np.ndarray
        The optimized density matrix.
    bloch_initial : Dict[Tuple[int, ...], np.ndarray]
        Initial Bloch vectors for each subsystem subset.
    bloch_final : Dict[Tuple[int, ...], np.ndarray]
        Final Bloch vectors for each subsystem subset.
    moments_initial : Dict[Tuple[int, ...], float]
        Initial moment norms for each subsystem subset.
    moments_final : Dict[Tuple[int, ...], float]
        Final moment norms for each subsystem subset.
    metric_name : str
        Name of the optimized metric.
    metric_initial : float
        Initial value of the metric.
    metric_final : float
        Final value of the metric.
    checks : Dict[str, Any]
        Additional checks and information about the optimization.
    optimizer_info : Dict[str, Any]
        Information about the optimization process.
    """
    rho_initial: np.ndarray
    rho_final: np.ndarray
    bloch_initial: Dict[Tuple[int, ...], np.ndarray]
    bloch_final: Dict[Tuple[int, ...], np.ndarray]
    moments_initial: Dict[Tuple[int, ...], float]
    moments_final: Dict[Tuple[int, ...], float]
    metric_name: str
    metric_initial: float
    metric_final: float
    checks: Dict[str, Any]
    optimizer_info: Dict[str, Any]

# ----------------------------------------
# Auxiliary functions
# ----------------------------------------

def choose_metric(d:int, subset_index_map: Dict[Tuple[int, ...], np.ndarray], metric: str) -> Tuple[Callable, Callable]:
    """
    Select the appropriate metric computation function based on the metric name.

    Parameters
    ----------
    metric : str
        The name of the metric to compute. Supported values are 'concurrence' and 'eof' (or variants).
    
    Returns
    -------
    Tuple[Callable, Callable, str]
        - compute_metric_opt: function used in optimization
        - compute_metric_report: function used for final reporting
    
    Raises
    ------
    ValueError
        If the metric is not supported.
    """
    is_two_qubit = (d == 4)
    is_bipartite = (list(subset_index_map.keys()) == 3)

    # --- Concurrence / EOF ---
    if metric == "concurrence":
        if not is_two_qubit:
            raise NotImplementedError("Concurrence optimization is only implemented for 2-qubit systems.")
        return compute_concurrence, compute_concurrence

    elif metric in {"eof", "entanglement_of_formation", "entanglement-of-formation"}:
        if not is_two_qubit:
            raise NotImplementedError("Entanglement of formation optimization is only implemented for 2-qubit systems.")
        return compute_concurrence, compute_eof

     # --- Trace norm / Negativity ---
    elif metric in {"trace_norm", "trace norm", "partial_trace_norm", "partial trace norm"}:
        if not is_bipartite:
            raise NotImplementedError("Trace norm optimization is only implemented for bipartite systems.")
        return compute_partial_trace_norm, compute_partial_trace_norm

    elif metric == "negativity":
        if not is_bipartite:
            raise NotImplementedError("Trace norm optimization is only implemented for bipartite systems.")
        return compute_partial_trace_norm, compute_negativity
    
    else:
        raise ValueError("Unsupported metric. Metric must be 'concurrence', 'eof', 'trace_norm' or 'negativity'.")

def trivial_result(d, tensor_basis: np.ndarray, subset_index_map: Dict[Tuple[int, ...], np.ndarray], rho: np.ndarray,
                   Rt: Dict[Tuple[int, ...], float], metric: str, result_message: str, result_success: bool = True) -> OptimizationResult:
    """
    Create an OptimizationResult for trivial cases where no optimization is required.

    Parameters
    ----------
    tensor_basis : np.ndarray
        Tensor-product operator basis.
    subset_index_map : Dict[Tuple[int, ...], np.ndarray]
        Mapping from subsystem subsets to indices in tensor_basis.
    rho : np.ndarray
        The density matrix for the result.
    Rt : Dict[Tuple[int, ...], float]
        Target Bloch vector norms.
    metric : str
        Name of the metric.
    reason : str
        Reason why the case is trivial.
    
    Returns
    -------
    OptimizationResult
        The result object with initial and final states set to the same.
    """
    
    _, compute_metric_report = choose_metric(d, subset_index_map, metric)
    
    r0 = compute_bloch_vector(tensor_basis, subset_index_map, rho)
    R0 = compute_bloch_norms_from_vector(r0)
    metric_value = compute_metric_report(rho)

    moments_distance = {}
    moments_equal = None
    for subset in subset_index_map.keys():
        moments_distance[subset] = float(abs(R0[subset] - Rt[subset]))
        moments_equal = np.allclose(np.array(list(R0.values())), np.array(list(R0.values())))
    
    moments_difference = np.mean([R0[subset] - Rt[subset] for subset in subset_index_map.keys()])

    return OptimizationResult(
        rho_initial=rho,
        rho_final=rho,
        bloch_initial=r0,
        bloch_final=r0,
        moments_initial=R0,
        moments_final=R0,
        metric_name=metric,
        metric_initial=metric_value,
        metric_final=metric_value,
        checks={"moments_difference": moments_difference, "moments_equal": moments_equal,
                "is_valid_dm": True,},
        optimizer_info={"mode": "trivial", "result_success": result_success, "result_message": result_message,},)

# ----------------------------------------
# Main optimizer
# ----------------------------------------

def optimize_moment_preserving_entanglement(d: int, tensor_basis: np.ndarray, subset_index_map: Dict[Tuple[int, ...], np.ndarray],
                                            Rt: Dict[Tuple[int, ...], float], optimization: str = "minimize",
                                            metric: str = "concurrence", cholesky_opt: bool = False,
                                            purity_tol: float = 1e-10, psd_tol: float = 1e-10, jac_tol: float = 1e-12,
                                            local_maxiter: int = 500) -> OptimizationResult:
    """
    Optimize entanglement while preserving Bloch moment norms for specified subsystems.
    
    Parameters
    ----------
    d : int
        Dimension of the quantum system.
    tensor_basis : np.ndarray
        Tensor-product operator basis, shape (k, dn, dn).
    subset_index_map : Dict[Tuple[int, ...], np.ndarray]
        Mapping from subsystem subsets to indices in tensor_basis.
    Rt : Dict[Tuple[int, ...], float]
        Target Bloch vector norms for each subsystem subset.
    metric : str, default="concurrence"
        The entanglement metric to optimize ('concurrence' or 'eof').
    optimization : str, default="minimize"
        Whether to 'minimize' or 'maximize' the metric.
    cholesky_opt : bool, default=False
        If True, use Cholesky parametrization for optimization.
    purity_tol : float, default=1e-10
        Tolerance for purity checks.
    psd_tol : float, default=1e-10
        Tolerance for positive semidefinite checks.
    local_maxiter : int, default=500
        Maximum iterations for the local optimizer.
    
    Returns
    -------
    OptimizationResult
        The result of the optimization.
    
    Raises
    ------
    ValueError
        If optimization mode is invalid.
    RuntimeError
        If initial state computation fails.
    """
    # --- 1. Find valid initial state ---
    compute_metric_opt, compute_metric_report = choose_metric(d, subset_index_map, metric)
    
    param_res: ParameterResult | None = None
    while param_res is None or not param_res.optimizer_info["success"]:
        param_res = compute_initial_param_repeat(d, tensor_basis, subset_index_map, Rt)
    
    x0, rho0, r0, R0 = param_res.param, param_res.rho, param_res.bloch, param_res.moments
    metric0 = compute_metric_report(rho0)
    
    # --- 2. Check trivial cases ---
    SR = sum([Rt[subset]**2 for subset in subset_index_map.keys()])
    
    if SR >= d - 1 - purity_tol:
        return trivial_result(
            d, tensor_basis, subset_index_map, rho0, Rt, metric,
            result_message="The input state is numerically pure. Entanglement cannot be improved."
        )

    if SR <= 0 + purity_tol:
        return trivial_result(
            d, tensor_basis, subset_index_map, np.identity(d, dtype=complex) / d, Rt, metric,
            result_message="The input state is numerically maximally mixed. Entanglement cannot be improved."
        )
    
    # --- 3. Convert to Cholesky parametrization ---
    if cholesky_opt:
        L = compute_cholesky(rho0, lower=True)
        x0 = compute_param_from_X(L, cholesky_opt)

    # --- 4. Shared cache for efficiency ---
    cache = {}

    def compute_shared(x: np.ndarray) -> Dict:
        """
        Compute and cache quantum state representations and derived quantities to avoid redundant calculations.
        
        Parameters
        ----------
        x : ndarray
            Optimization parameter vector representing the state in Cholesky form.
        
        Returns
        -------
        Dict
            Cache dictionary containing:
            - 'x' : ndarray
                Copy of the input parameter vector.
            - 'X' : ndarray
                Cholesky decomposition matrix computed from parameters.
            - 'rho' : ndarray
                Density matrix (2D array) derived from X.
            - 'r' : ndarray
                Bloch vector (1D array) computed via tensor basis contraction with rho.
            - 'R' : ndarray
                Bloch norms computed from the Bloch vector r.
            - 'jac' : None
                Placeholder for Jacobian (reset on cache update).
        """
        if "x" not in cache or not np.array_equal(x, cache["x"]):
            X = compute_X_from_param(x, cholesky_opt)
            rho = compute_dm_from_X(X)
            r = compute_bloch_vector(tensor_basis, subset_index_map, rho)
            R = compute_bloch_norms_from_vector(r)

            cache["x"] = x.copy()
            cache["X"] = X
            cache["rho"] = rho
            cache["r"] = r
            cache["R"] = R
            cache["jac"] = None

        return cache
    
    # --- 5. Objective function ---
    def compute_objective(optimization: str):
        """
        Compute the objective function for optimization.
        It gives a flexible output deppending if maximizing or minimizing.
        
        Parameters
        ----------
        x : array_like
            Optimization variables representing parameters to be optimized: parametrization of the density matrix.
        
        Returns
        -------
        float
            The objective function value.
            Returns the negative of the metric when maximizing, or the metric directly when minimizing.
        """
        if optimization == "maximize":
            def objective(x: np.ndarray) -> float:
                rho = compute_shared(x)["rho"]
                return -compute_metric_opt(rho)
        elif optimization == "minimize":
            def objective(x: np.ndarray) -> float:
                rho = compute_shared(x)["rho"]
                return compute_metric_opt(rho)
        else:
            raise ValueError(f"Input 'optimization' must be either 'maximize' or 'minimize', got {optimization}")
        return objective
    
    objective = compute_objective(optimization)
    
    # --- 6. Constraints and Jacobian (one per subsystem) ---
    def compute_jacobian(x: np.ndarray) -> np.ndarray:
        """
        Calculates the Jacobian of the objective function.
        Results are cached in the shared data dictionary to avoid redundant computations.

        Parameters
        ----------
        x : np.ndarray
            Flattened array of optimization variables, containing the density matrix parametrization.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape (n_constraints, n_variables).
            Each row corresponds to the gradient of one constraint with respect to all variables.
        """
        data = compute_shared(x)

        if data["jac"] is not None:
            return data["jac"]

        X = data["X"]
        rho = data["rho"]
        r = data["r"]

        XX = X @ X.conj().T
        trXX = np.trace(XX).real

        jac_rows = []

        for subset, indices in subset_index_map.items():
            r_M = r[subset]
            norm = np.linalg.norm(r_M)

            if norm > jac_tol:
                direction = r_M / norm
                grad_rho = np.tensordot(direction, tensor_basis[indices], axes=1)
                scalar = np.real(np.trace(grad_rho @ rho))
                grad_X = (2 / trXX) * (grad_rho @ X - scalar * X)
            else:
                grad_X = np.zeros_like(X)

            if cholesky_opt:
                lower_indices = np.tril_indices(X.shape[0])
                grad_vals = grad_X[lower_indices]
                grad = np.empty(2 * len(grad_vals), dtype=float)
            else:
                grad_vals = grad_X.ravel()
                grad = np.empty_like(x)

            grad[0::2] = np.real(grad_vals)
            grad[1::2] = np.imag(grad_vals)

            jac_rows.append(grad)

        data["jac"] = np.vstack(jac_rows)
        return data["jac"]

    constraints = []
    def compute_constraints_fun(subset):
        target = Rt[subset]
        def fun(x):
            R = compute_shared(x)["R"]
            return R[subset] - target
        return fun

    def compute_constraints_jac(i):
        def jac(x):
            return compute_jacobian(x)[i]
        return jac

    for i, subset in enumerate(subset_index_map.keys()):
        constraints.append({
            "type": "eq",
            "fun": compute_constraints_fun(subset),
            "jac": compute_constraints_jac(i),
        })
    
    # --- 7. Run optimization ---
    result = minimize(objective, x0,
        method="SLSQP", constraints=constraints,
        options={
            "maxiter": local_maxiter,
            "ftol": 1e-12,
            "disp": False
        }
    )
    
    # --- 8. Extract final state ---
    cache_result = compute_shared(result.x)
    rho_opt, r_opt, R_opt = cache_result["rho"], cache_result["r"], cache_result["R"]
    metric_opt = compute_metric_report(rho_opt)

    # --- 9. Run checks on the solution ---
    moments_distance = {}
    moments_equal = None
    for subset in subset_index_map.keys():
        moments_distance[subset] = float(abs(R_opt[subset] - Rt[subset]))
        moments_equal = np.allclose(np.array(list(R0.values())), np.array(list(R_opt.values())))
    
    is_valid_dm, is_valid_dm_info = compute_is_valid_dm(rho_opt, psd_tol)
    
    # --- 10. Final output ---
    checks = {
        "moments_distance": moments_distance,
        "moments_equal": moments_equal,
        "is_valid_dm": is_valid_dm,
        "is_valid_dm_info": is_valid_dm_info,
    }

    optimizer_info = {
        "mode": "moment_preserving_bloch",
        "metric_optimized": metric,
        "cache_mode": "exact",
        "result_success": bool(result.success),
        "result_message": str(result.message),
        "fun": complex(result.fun),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
    }

    return OptimizationResult(
        rho_initial=rho0,
        rho_final=rho_opt,
        bloch_initial=r0,
        bloch_final=r_opt,
        moments_initial=R0,
        moments_final=R_opt,
        metric_name=metric,
        metric_initial=metric0,
        metric_final=metric_opt,
        checks=checks,
        optimizer_info=optimizer_info,
        )