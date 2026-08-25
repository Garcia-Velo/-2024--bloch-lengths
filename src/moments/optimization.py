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
from typing import Callable, Any

# Local importations
from moments.bloch import compute_bloch_vector, compute_bloch_norms_from_vector
from moments.initialization import ParameterResult, compute_initial_param_repeat, compute_param_from_X, compute_X_from_param, compute_dm_from_X
from moments.ent_meas import compute_concurrence, compute_eof, compute_tr_norm, compute_pt, compute_pt_norm, compute_pt_norm_jac, compute_negativity
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
    bloch_initial: dict[tuple[int, ...], np.ndarray]
    bloch_final: dict[tuple[int, ...], np.ndarray]
    moments_initial: dict[tuple[int, ...], float]
    moments_final: dict[tuple[int, ...], float]
    metric_name: str
    metric_initial: float
    metric_final: float
    checks: dict[str, Any]
    optimizer_info: dict[str, Any]

@dataclass
class MetricInputs:
    """
    Bundles the data needed to evaluate an entanglement metric (and, where available, its Jacobian).
    All metric adapters returned by choose_metric take a single MetricInputs argument, even though each metric only reads a subset of its fields.
    """
    dim: list[int]
    A: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None

# ----------------------------------------
# Auxiliary functions
# ----------------------------------------

def choose_metric(dim: list[int], metric: str) -> tuple[Callable, Callable, None | Callable]:
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
    # Concurrence-based metrics are only supported for two qubits.
    is_two_qubit = (len(dim) == 2 and dim[0] == dim[1] == 2)

    # --- Concurrence / EOF ---
    if metric == "concurrence":
        # Reject unsupported dimensions before creating metric callbacks.
        if not is_two_qubit:
            raise NotImplementedError("Concurrence optimization is only implemented for 2-qubit systems.")
        def compute_metric_report(inputs: MetricInputs) -> float:
            # Evaluate concurrence from the density matrix.
            if inputs.A is not None:
                return compute_concurrence(inputs.A)
            else:
                raise ValueError("Must provide A.")
        return compute_metric_report, compute_metric_report, None

    elif metric == "entanglement_of_formation":
        # Entanglement of formation is implemented here for two qubits only.
        if not is_two_qubit:
            raise NotImplementedError("Entanglement of formation optimization is only implemented for 2-qubit systems.")
        def compute_metric_report(inputs: MetricInputs) -> float:
            # Compute the final metric value for reporting.
            if inputs.A is not None:
                return float(compute_eof(rho=inputs.A))
            else:
                raise ValueError("Must provide A.")
        def compute_metric_opt(inputs: MetricInputs) -> float:
            # Optimize concurrence because it is monotonic with this metric.
            if inputs.A is not None:
                return compute_concurrence(inputs.A)
            else:
                raise ValueError("Must provide A.")
        return compute_metric_report, compute_metric_opt, None
    
     # --- Trace norm / Negativity ---
    elif metric == "partial_trace_norm":
        def compute_metric_report(inputs: MetricInputs) -> float:
            # Use precomputed eigenvalues when available to avoid repeated work.
            if inputs.eigenvalues is not None:
                return compute_pt_norm(inputs.dim, eigenvalues=inputs.eigenvalues)
            return compute_pt_norm(inputs.dim, A=inputs.A)
        def compute_metric_jac(inputs: MetricInputs) -> np.ndarray:
            # Compute the corresponding analytical Jacobian.
            if inputs.eigenvalues is not None:
                return compute_pt_norm_jac(inputs.dim, eigenvalues=inputs.eigenvalues, eigenvectors=inputs.eigenvectors)
            return compute_pt_norm_jac(inputs.dim, A=inputs.A)
        return compute_metric_report, compute_metric_report, compute_metric_jac
    elif metric == "negativity":
        def compute_metric_report(inputs: MetricInputs) -> float:
            # Convert the partial-transpose trace norm into negativity.
            if inputs.eigenvalues is not None:
                A_pt = compute_tr_norm(eigenvalues=inputs.eigenvalues)
                return float(compute_negativity(trace_norm_pt=A_pt))
            return float(compute_negativity(rho=inputs.A, dim=inputs.dim))
    
        def compute_metric_opt(inputs: MetricInputs) -> float:
            # Optimize the proportional partial-transpose norm.
            if inputs.eigenvalues is not None:
                return compute_pt_norm(inputs.dim, eigenvalues=inputs.eigenvalues)
            return compute_pt_norm(inputs.dim, A=inputs.A)
    
        def compute_metric_jac(inputs: MetricInputs) -> np.ndarray:
            # Scale factor lives here, once.
            # Build the Jacobian of the partial-transpose norm.
            if inputs.eigenvalues is not None:
                G = compute_pt_norm_jac(inputs.dim, eigenvalues=inputs.eigenvalues, eigenvectors=inputs.eigenvectors)
            else:
                G = compute_pt_norm_jac(inputs.dim, A=inputs.A)
            return 0.5 * G
        return compute_metric_report, compute_metric_opt, compute_metric_jac
    
    else:
        raise ValueError("Unsupported metric. Metric must be 'concurrence', 'entanglement_of_formation', 'partial_trace_norm' or 'negativity'.")

def trivial_result(dim, tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray], rho: np.ndarray,
                   Rt: dict[tuple[int, ...], float], metric: str, result_message: str, result_success: bool = True) -> OptimizationResult:
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
    # Select the reporting function.
    compute_metric_report, _, _ = choose_metric(dim, metric)
    
    # Compute the Bloch vectors and their norms for the unchanged state.
    r0 = compute_bloch_vector(tensor_basis, subset_index_map, rho)
    R0 = compute_bloch_norms_from_vector(r0)
    # Evaluate the metric once because initial and final states are identical.
    metric_value = compute_metric_report(MetricInputs(dim=dim, A=rho))

    
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

def compute_chain_rule_grad(d: int, grad_rho: np.ndarray, data: dict, cholesky_opt: bool) -> np.ndarray:
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
        # Retrieve the cached parametrization and normalized density matrix.
        X, rho, tau = data["X"], data["rho"], data["tau"]

        # Remove the trace-direction component before mapping the gradient to X.
        scalar = np.real(np.trace(grad_rho @ rho))
        grad_X = (2.0 / tau) * (grad_rho - scalar * np.eye(d, dtype=complex)) @ X

        if cholesky_opt:
            # Keep only independent lower-triangular Cholesky entries.
            lower_idx = np.tril_indices(d)
            grad_vals = grad_X[lower_idx]
        else:
            grad_vals = grad_X.ravel()

        # Interleave real and imaginary parts for scipy's real parameter vector.
        grad = np.empty(2 * len(grad_vals), dtype=float)
        grad[0::2] = np.real(grad_vals)
        grad[1::2] = np.imag(grad_vals)

        return grad

# ----------------------------------------
# Main optimizer
# ----------------------------------------

def opt_moment_preserving_ent(dim: list[int], tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray],
                                            Rt: dict[tuple[int, ...], float], optimization: str = "minimize",
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
    metric : str, default="negativity"
        The entanglement metric to optimize.
    optimization : str, default="minimize"
        Whether to 'minimize' or 'maximize' the metric.
    cholesky_opt : bool, default=True
        If True, use Cholesky parametrization for optimization.
    exact_jac : bool, default=False
        If True, compute the exact Jacobian for the trace norm metric.
    purity_tol : float, default=1e-10
        Tolerance for purity checks.
    psd_tol : float, default=1e-10
        Tolerance for positive semidefinite checks.
    jac_tol : float, default=1e-10
            Tolerance for Jacobian calculations.
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
    # Derive the subsystem count and total Hilbert-space dimension.
    N = len(dim)
    d = int(np.prod(dim))
    
    if N <= 1:
        raise ValueError(f"The number of subsystems must be greater or equal than 1.")
    elif N > 2:
        raise NotImplementedError("Optimization is only implemented for bipartite systems.")
    
    # Resolve all metric callbacks before constructing the optimizer.
    compute_metric_report, compute_metric_opt, compute_metric_jac = choose_metric(dim, metric)
    
    # --- 1. Find valid initial state ---
    # Retry initialization until a valid starting state is produced.
    param_res: ParameterResult | None = None
    while param_res is None or not param_res.optimizer_info["success"]:
        param_res = compute_initial_param_repeat(d, tensor_basis, subset_index_map, Rt)
    
    # Unpack the starting parameters, state, Bloch vectors, and moments.
    x0, rho0, r0, R0 = param_res.param, param_res.rho, param_res.bloch, param_res.moments
    metric0 = compute_metric_report(MetricInputs(dim=dim, A=rho0))

    # --- 2. Check trivial cases ---
    # Use total squared moment length to detect trivial boundary cases.
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
        # Factor the initial state so trial states stay positive semidefinite.
        L = compute_cholesky(rho0, lower=True)
        x0 = compute_param_from_X(L, cholesky_opt)

    # --- 4. Shared cache for efficiency ---
    # Reuse expensive state calculations across optimizer callbacks.
    cache: dict = {}

    def compute_shared(x: np.ndarray) -> dict:
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
            # Rebuild cached quantities only when the parameter vector changes.
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
                # Cache partial-transpose eigendata for exact metric derivatives.
                rho_pt = compute_pt(dim, rho)
                eigvals, eigvecs = eigh(rho_pt)
                metric_inputs = MetricInputs(dim=dim, eigenvalues=eigvals, eigenvectors=eigvecs)

                assert compute_metric_jac is not None
                cache.update({
                    "metric": compute_metric_opt(metric_inputs),
                    "S_pt": compute_metric_jac(metric_inputs),
                    })
        
        return cache
    
    # --- 5. Objective function ---
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
        # Negate the metric for maximization because scipy minimizes objectives.
        sign = -1.0 if optimization == "maximize" else 1.0

        # Reject unsupported optimization directions before creating the callback.
        if optimization not in {"maximize", "minimize"}:
            raise ValueError(f"Input 'optimization' must be 'maximize' or 'minimize', got '{optimization}'.")
        
        # Use cached metric data when the analytical Jacobian is enabled.
        if exact_jac:
            def objective(x: np.ndarray) -> float:
                return sign * compute_shared(x)["metric"]
        else:
            def objective(x: np.ndarray) -> float:
                # Reconstruct the density matrix and evaluate the metric directly.
                rho = compute_shared(x)["rho"]
                return sign * compute_metric_opt(MetricInputs(dim=dim, A=rho))
        
        return objective
    
    # Build the objective callback with the requested optimization direction.
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
        # Reuse all state quantities computed for this parameter vector.
        data = compute_shared(x)

        # Return the cached Jacobian when another constraint requests it.
        if data["jac_constraints"] is not None:
            return data["jac_constraints"]
        
        # Read the Bloch vectors and prepare one gradient row per constraint.
        r = data["r"]
        rows = []

        # Different subsystem subsets define separate moment-norm constraints.
        for subset, indices in subset_index_map.items():
            r_M = r[subset]
            norm = np.linalg.norm(r_M)

            # Normalize the vector to obtain the derivative of its norm.
            if norm > jac_tol:
                direction = r_M / norm
                # Contract the direction with the operator basis to get dR/drho.
                grad_rho = np.tensordot(direction, tensor_basis[indices], axes=1)
            else:
                # Use a zero derivative at a numerically vanishing norm.
                grad_rho = np.zeros((d, d), dtype=complex)

            # Convert the matrix gradient into the optimizer's real parameters.
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
        # Reuse the state and partial-transpose data for this parameter vector.
        data = compute_shared(x)

        # Avoid recomputing the metric gradient during the same optimizer step.
        if data["jac_metric"] is not None:
            return data["jac_metric"]
        
        # Apply the same sign convention as the objective callback.
        sign = -1.0 if optimization == "maximize" else 1.0
        grad = sign * compute_chain_rule_grad(d, data["S_pt"], data, cholesky_opt)

        data["jac_metric"] = grad

        return grad

    # --- 9. Constraints ---
    # Create a callback that measures deviation from one target moment.
    def _make_constraint_fun(subset):
        target = Rt[subset]
        def fun(x):
            return compute_shared(x)["R"][subset] - target
        return fun
    
    # Create a callback that selects one row of the constraint Jacobian.
    def _make_constraint_jac(i):
        def jac(x):
            return compute_jac_constraints(x)[i]
        return jac
    
    # Build scipy's list of equality constraints in subset order.
    constraints = [
        {
            "type": "eq",
            "fun": _make_constraint_fun(subset),
            "jac": _make_constraint_jac(i),
        }
        for i, subset in enumerate(subset_index_map)
    ]
    
    # --- 10. Run optimization ---
    # Configure SLSQP with the equality constraints and convergence settings.
    minimize_kwargs = {
        "method": "SLSQP",
        "constraints": constraints,
        "options": {"maxiter": local_maxiter, "ftol": 1e-12, "disp": False},
    }
    # Exact derivatives are available only for metrics with a Jacobian adapter.
    if exact_jac and compute_metric_jac is None:
        raise NotImplementedError(f"Exact Jacobian is not implemented for metric '{metric}'. Use exact_jac=False instead.")
    # Pass the analytical objective Jacobian when requested.
    if exact_jac:
        minimize_kwargs["jac"] = compute_jac_metric
    # Execute the constrained local optimization from the initialized parameters.
    result = minimize(objective, x0, **minimize_kwargs)

    # --- 11. Extract final state ---
    # Compute and cache all representations at the optimizer's final parameters.
    cache_result = compute_shared(result.x)
    rho_opt, r_opt, R_opt = cache_result["rho"], cache_result["r"], cache_result["R"]
    metric_opt = compute_metric_report(MetricInputs(dim=dim, A=rho_opt))

    # --- 12. Run checks on the solution ---
    # Measure the absolute error for every preserved moment norm.
    moments_distance = {subset: float(abs(R_opt[subset] - Rt[subset])) for subset in subset_index_map}
    moments_equal = np.allclose(np.array(list(R0.values())), np.array(list(R_opt.values())))
    
    # Verify that the optimized matrix is still a valid density matrix.
    is_valid_dm, is_valid_dm_info = compute_is_valid_dm(rho_opt, psd_tol)
    
    # --- 13. Final output ---
    # Collect numerical checks for callers and downstream reporting.
    checks = {
        "moments_distance": moments_distance,
        "moments_equal": moments_equal,
        "is_valid_dm": is_valid_dm,
        "is_valid_dm_info": is_valid_dm_info,
    }

    # Record the optimizer configuration and scipy's termination statistics.
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

    # Return the initial and optimized states together with all diagnostics.
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