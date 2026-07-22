#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np
from scipy.linalg import cholesky as compute_cholesky
from scipy.linalg import eigh
from scipy.optimize import minimize

# Auxiliary python functions
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Any

# Local importations
from moments.bloch import compute_bloch_vector, compute_bloch_norms_from_vector
from moments.initialization import ParameterResult, compute_initial_param_repeat, compute_param_from_X, compute_X_from_param, compute_dm_from_X
from moments.ent_meas import compute_concurrence, compute_eof, compute_pt, compute_pt_norm, compute_pt_norm_jac, compute_negativity
from moments.quantum import compute_is_valid_dm

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

def choose_metric(dim: list[int], metric: str) -> Tuple[Callable, Callable, None | Callable]:
    """
    Select the appropriate metric computation function based on the metric name.

    Parameters
    ----------
    dim : list[int]
        List of local dimensions of the quantum system.
    metric : str
        The name of the metric to compute. Supported values are 'concurrence' and 'eof' (or variants).
    
    Returns
    -------
    Tuple[Callable, Callable, Callable]
        - compute_metric_report: function used for final reporting
        - compute_metric_opt: function used in optimization
        - compute_metric_jac: function used for computing the Jacobian of the metric

    Raises
    ------
    ValueError
        If the metric is not supported.
    """
    is_two_qubit = (len(dim) == 2 and dim[0] == dim[1] == 2)

    # --- Concurrence / EOF ---
    if metric == "concurrence":
        if not is_two_qubit:
            raise NotImplementedError("Concurrence optimization is only implemented for 2-qubit systems.")
        # return compute_concurrence, compute_concurrence, compute_concurrence_jac
        return compute_concurrence, compute_concurrence, None

    elif metric == "entanglement_of_formation":
        if not is_two_qubit:
            raise NotImplementedError("Entanglement of formation optimization is only implemented for 2-qubit systems.")
        # return compute_eof, compute_concurrence, compute_concurrence_jac
        return compute_eof, compute_concurrence, None

     # --- Trace norm / Negativity ---
    elif metric == "partial_trace_norm":
        return compute_pt_norm, compute_pt_norm, compute_pt_norm_jac

    elif metric == "negativity":
        return compute_negativity, compute_pt_norm, compute_pt_norm_jac
    
    else:
        raise ValueError("Unsupported metric. Metric must be 'concurrence', 'entanglement_of_formation', 'partial_trace_norm' or 'negativity'.")

def trivial_result(dim, tensor_basis: np.ndarray, subset_index_map: Dict[Tuple[int, ...], np.ndarray], rho: np.ndarray,
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
    compute_metric_report, _, _ = choose_metric(dim, metric)
    
    r0 = compute_bloch_vector(tensor_basis, subset_index_map, rho)
    R0 = compute_bloch_norms_from_vector(r0)
    metric_value = compute_metric_report(dim, A=rho)
    
    moments_equal = np.allclose(np.array(list(R0.values())), np.array(list(Rt.values())))
    moments_difference = float(np.mean([R0[subset] - Rt[subset] for subset in subset_index_map]))

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
        checks={
            "moments_difference": moments_difference,
            "moments_equal": moments_equal,
            "is_valid_dm": True,
            },
        optimizer_info={
            "mode": "trivial",
            "result_success": result_success,
            "result_message": result_message,
            },
        )

def compute_chain_rule_grad(d: int, grad_rho: np.ndarray, data: Dict, cholesky_opt: bool) -> np.ndarray:
        """
        Map a gradient w.r.t. rho to a gradient w.r.t. the parameter vector x.

        Implements the common chain rule
            df/dx  =  Re/Im parts of  (2/tau) * (grad_rho - Tr(grad_rho @ rho)*I) @ X
        shared by both the constraint Jacobian and the metric Jacobian.

        Parameters
        ----------
        grad_rho : np.ndarray
            Gradient of a scalar f w.r.t. rho, shape (d, d), complex Hermitian.
        data : Dict
            Populated cache dictionary (must contain 'X', 'rho', 'tau').

        Returns
        -------
        np.ndarray
            Real gradient vector of f w.r.t. x, same length as x.
        """
        X, rho, tau = data["X"], data["rho"], data["tau"]

        scalar = np.real(np.trace(grad_rho @ rho))
        grad_X = (2.0 / tau) * (grad_rho - scalar * np.eye(d, dtype=complex)) @ X

        if cholesky_opt:
            lower_idx = np.tril_indices(d)
            grad_vals = grad_X[lower_idx]
        else:
            grad_vals = grad_X.ravel()

        grad = np.empty(2 * len(grad_vals), dtype=float)
        grad[0::2] = np.real(grad_vals)
        grad[1::2] = np.imag(grad_vals)

        return grad

# ----------------------------------------
# Main optimizer
# ----------------------------------------

def opt_moment_preserving_ent(dim: list[int], tensor_basis: np.ndarray, subset_index_map: Dict[Tuple[int, ...], np.ndarray],
                                            Rt: Dict[Tuple[int, ...], float], optimization: str = "minimize",
                                            metric: str = "negativity", cholesky_opt: bool = True, exact_jac: bool = False,
                                            purity_tol: float = 1e-10, psd_tol: float = 1e-10, jac_tol: float = 1e-10,
                                            local_maxiter: int = 500) -> OptimizationResult:
    """
    Optimize entanglement while preserving Bloch moment norms for specif`ied subsystems.
    
    Parameters
    ----------
    dim : list[int]
        List of local dimensions of the quantum system.
    tensor_basis : np.ndarray
        Tensor-product operator basis, shape (k, dn, dn).
    subset_index_map : Dict[Tuple[int, ...], np.ndarray]
        Mapping from subsystem subsets to indices in tensor_basis.
    Rt : Dict[Tuple[int, ...], float]
        Target Bloch vector norms for each subsystem subset.
    x0_warm: np.ndarray | None, defalt=None
        Guess of an initial state
    metric : str, default="concurrence"
        The entanglement metric to optimize ('concurrence' or 'eof').
    optimization : str, default="minimize"
        Whether to 'minimize' or 'maximize' the metric.
    cholesky_opt : bool, default=False
        If True, use Cholesky parametrization for optimization.
    exact_jac : bool, default=False
        If True, compute the exact Jacobian for the trace norm metric.
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
    # --- 0. Resolve dimensions ---
    N = len(dim)
    d = int(np.prod(dim))
    
    if N <= 1:
        raise ValueError(f"The number of subsystems must be greater or equal than 2.")
    elif N > 2:
        raise NotImplementedError("Optimization is only implemented for bipartite systems.")
    
    compute_metric_report, compute_metric_opt, compute_metric_jac = choose_metric(dim, metric)
    
    # --- 1. Find valid initial state ---
    param_res: ParameterResult | None = None
    while param_res is None or not param_res.optimizer_info["success"]:
        param_res = compute_initial_param_repeat(d, tensor_basis, subset_index_map, Rt)
    
    x0, rho0, r0, R0 = param_res.param, param_res.rho, param_res.bloch, param_res.moments
    metric0 = compute_metric_report(dim, A=rho0)

    # --- 2. Check trivial cases ---
    SR = sum([Rt[subset]**2 for subset in subset_index_map.keys()])
    
    if SR >= d - 1 - purity_tol:
        return trivial_result(
            dim, tensor_basis, subset_index_map, rho0, Rt, metric,
            result_message="The input state is numerically pure. Entanglement cannot be improved."
        )

    if SR <= 0 + purity_tol:
        return trivial_result(
            dim, tensor_basis, subset_index_map, np.identity(d, dtype=complex) / d, Rt, metric,
            result_message="The input state is numerically maximally mixed. Entanglement cannot be improved."
        )
    
    # --- 3. Convert to Cholesky parametrization ---
    if cholesky_opt:
        L = compute_cholesky(rho0, lower=True)
        x0 = compute_param_from_X(L, cholesky_opt)

    # --- 4. Shared cache for efficiency ---
    cache: Dict = {}

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
            Cache containing:
            - 'x' : np.ndarray
                Input parameter vector.
            - 'X' : np.ndarray
                Cholesky factor.
            - 'rho' : np.ndarray
                Density matrix.
            - 'r' : np.ndarray
                Bloch vectors per subset.
            - 'R' : np.ndarray
                Bloch norms per subset.
            - 'tau': float
                Tr(XX^†), used in chain-rule kernel.
            - 'jac_constraints' : np.ndarray | None
                Cached constraint Jacobian rows.
            - 'jac_metric' : np.ndarray | None
                Cached metric Jacobian row.
            [Only when use_pt_metric is True:]
            - 'eigvals' : np.ndarray
                Eigenvalues of rho^Gamma.
            - 'S_pt' : np.ndarray
                [sgn(rho^Gamma)]^Gamma.
            - 'trace_norm_pt' : float
                ||rho^Gamma||_1 = sum(|eigvals|).
        """
        if "x" not in cache or not np.array_equal(x, cache["x"]):
            X = compute_X_from_param(x, cholesky_opt)
            rho = compute_dm_from_X(X)
            r = compute_bloch_vector(tensor_basis, subset_index_map, rho)
            R = compute_bloch_norms_from_vector(r)
            tau = np.real(np.trace(X @ X.conj().T))

            cache.update({
                "x": x.copy(),
                "X": X,
                "rho": rho,
                "r": r,
                "R": R,
                "tau": tau,
                "jac_constraints": None,
                "jac_metric": None,
            })

            if exact_jac:
                rho_pt = compute_pt(dim, rho)
                eigvals, eigvecs = eigh(rho_pt)
                metric = compute_metric_opt(dim, eigenvalues=eigvals)
                jac = compute_metric_jac(dim, eigenvalues=eigvals, eigenvectors=eigvecs)

                cache.update({
                    "metric": metric,
                    "S_pt": jac,
                })
        
        return cache
    
    # --- 6. Objective function ---
    def compute_objective(optimization: str):
        """
        Build the objective function for optimization.
        It gives a flexible output deppending if maximizing or minimizing.
        
        Parameters
        ----------
        optimization : str
            'minimize' or 'maximize', depending on the desired optimization.
        
        Returns
        -------
        Callable
            Objective function. Returns the negative of the metric when maximizing, or the metric directly when minimizing.
        """
        sign = -1.0 if optimization == "maximize" else 1.0

        if optimization not in {"maximize", "minimize"}:
            raise ValueError(f"Input 'optimization' must be 'maximize' or 'minimize', got '{optimization}'.")
        
        if exact_jac:
            def objective(x: np.ndarray) -> float:
                # Reuse the trace norm already computed during cache update.
                return sign * compute_shared(x)["metric"]
        else:
            def objective(x: np.ndarray) -> float:
                return sign * compute_metric_opt(compute_shared(x)["rho"])
        
        return objective
    
    objective = compute_objective(optimization)
    
    # --- 7. Constraint Jacobian ---
    def compute_jac_constraints(x: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of the moment-norm constraints. One row per subsystem subset.
        Results are cached to avoid redundant computations within the same optimizer step.

        Parameters
        ----------
        x : np.ndarray
            Optimization parameter vector, density matrix parametrization.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape (n_subsets, len(x)).
        """
        data = compute_shared(x)

        if data["jac_constraints"] is not None:
            return data["jac_constraints"]
        
        r = data["r"]
        rows = []

        for subset, indices in subset_index_map.items():
            r_M = r[subset]
            norm = np.linalg.norm(r_M)

            if norm > jac_tol:
                direction = r_M / norm
                grad_rho = np.tensordot(direction, tensor_basis[indices], axes=1)
            else:
                grad_rho = np.zeros((d, d), dtype=complex)

            rows.append(compute_chain_rule_grad(d, grad_rho, data, cholesky_opt))

        data["jac_constraints"] = np.vstack(rows)
        return data["jac_constraints"]

    # --- 8. Metric Jacobian (for partial-transpose metrics) ---
    def compute_jac_metric(x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian of the objective w.r.t. the parameter vector x.
        The sign matrix and its back-transpose are computed once inside compute_shared and reused here.
        The chain rule is then applied via compute_chain_rule_grad.

        Parameters
        ----------
        x : np.ndarray
            Optimization parameter vector, density matrix parametrization.

        Returns
        -------
        np.ndarray
            Gradient vector of the objective w.r.t. x, same length as x.
        """
        data = compute_shared(x)

        if data["jac_metric"] is not None:
            return data["jac_metric"]
        
        # G = 0.5 * data["S_pt"]
        G = data["S_pt"]
        sign = -1.0 if optimization == "maximize" else 1.0
        grad = sign * compute_chain_rule_grad(d, G, data, cholesky_opt)

        data["jac_metric"] = grad

        if metric == "negativity":
            return 0.5 * grad
        else:
            return grad
    
    # --- 9. Constraints ---
    def _make_constraint_fun(subset):
        target = Rt[subset]
        def fun(x):
            return compute_shared(x)["R"][subset] - target
        return fun
    
    def _make_constraint_jac(i):
        def jac(x):
            return compute_jac_constraints(x)[i]
        return jac
    
    constraints = [
        {
            "type": "eq",
            "fun": _make_constraint_fun(subset),
            "jac": _make_constraint_jac(i),
        }
        for i, subset in enumerate(subset_index_map)
    ]
    
    # --- 10. Run optimization ---
    minimize_kwargs = {
        "method": "SLSQP",
        "constraints": constraints,
        "options": {"maxiter": local_maxiter, "ftol": 1e-12, "disp": False},
    }
    if exact_jac:
        minimize_kwargs["jac"] = compute_jac_metric
    result = minimize(objective, x0, **minimize_kwargs)

    # --- 10. Extract final state ---
    cache_result = compute_shared(result.x)
    x_opt, rho_opt, r_opt, R_opt = cache_result["x"], cache_result["rho"], cache_result["r"], cache_result["R"]
    metric_opt = compute_metric_report(dim, A=rho_opt)

    # --- 11. Run checks on the solution ---
    moments_distance = {subset: float(abs(R_opt[subset] - Rt[subset])) for subset in subset_index_map}
    moments_equal = np.allclose(np.array(list(R0.values())), np.array(list(R_opt.values())))
    
    is_valid_dm, is_valid_dm_info = compute_is_valid_dm(rho_opt, psd_tol)
    
    # --- 12. Final output ---
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