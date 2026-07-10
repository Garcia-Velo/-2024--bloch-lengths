#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np
import cvxpy as cp

# Auxiliary python functions
from typing import NamedTuple, Optional, Sequence
from cvxpy.constraints.constraint import Constraint

# Local importations
from moments.quantum import compute_is_valid_dm
from moments.ppt_mixer.utilities import enumerate_bipartitions, compute_pt_cvxpy

# ----------------------------------------
# Definitions
# ----------------------------------------


class DecomposableWitnessResult(NamedTuple):
    """
    Attributes
    ----------
    minval : float
        Optimal value of ``tr(W rho)`` over all fully decomposable witnesses ``W``. Negative if ``rho`` is genuinely multipartite
        entangled.
    witness : np.ndarray or None
        The optimal witness ``W`` (as a ``(d, d)`` Hermitian complex array) if ``minval < -precision``, i.e. if entanglement was
        detected. Otherwise a ``(d, d)`` zero matrix.
    """

    minval: float
    witness: np.ndarray

# ----------------------------------------
# Decomposable witness optimization
# ----------------------------------------

def find_decomposable_witness(
        dim: Sequence[int],
        rho: np.ndarray,
        solver: Optional[str] = None,
        precision: float = 1e-12,
        verbose: bool = False,
        solver_kwargs: Optional[dict] = None,
        ) -> DecomposableWitnessResult:
    """
    Solves the semidefinite program of eq. (4) in Phys. Rev. Lett. 106, 190502 (2011).
    Test whether ``rho`` is a PPT mixture or whether it is GME by finding the fully decomposable witness minimising ``tr(W rho)``.

    Parameters
    ----------
    dim : sequence of intpr
        Local Hilbert space dimension of each of the ``N`` subsystems, with ``d = prod(dim)``.
    rho : np.ndarray
        ``(d, d)`` array describing the state.
    solver : str, optional
        Name of the CVXPY-registered conic solver to use (e.g. ``"SCS"``, ``"CVXOPT"``, ``"MOSEK"``).
        If ``None``, CVXPY chooses an installed solver automatically (typically SCS).
    precision : float, optional
        Numerical threshold below which the optimal value is considered strictly negative (i.e. entanglement detected).
    verbose : bool, optional
        If True, print solver progress output.
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to ``cvxpy.Problem.solve`` (e.g. ``{"eps": 1e-8}`` to tighten SCS's convergence tolerance).

    Returns
    -------
    DecomposableWitnessResult
        Named tuple ``(minval, witness)``, see :class:`DecomposableWitnessResult`.

    Raises
    ------
    ValueError
        If ``rho`` is not a valid quantum state, or if ``dim`` is inconsistent with ``rho``.
    RuntimeError
        If the SDP solver fails to reach an optimal (or optimal inaccurate) solution.
    """
    # ---------- Validations ----------
    # Hilbert space dimension
    d = int(np.prod(dim))
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("`rho` must be a square matrix.")
    if rho.shape != (d, d):
        raise ValueError("`dim` is incompatible with `rho`.")
    if np.min(dim) < 2:
        raise ValueError("All local dimensions must be at least 2.")
    
    # Density matrix
    is_valid, _ = compute_is_valid_dm(rho=rho, tol=precision)
    if not is_valid:
        raise ValueError(f"`rho` is not Hermitian within tolerance {precision} ")
    
    # ---------- Definitions ----------
    # Compute all possible bipartitions of {1, ..., N}
    N = len(dim)
    bipartitions = enumerate_bipartitions(N)

    # Define arguments for the solver
    solver_kwargs = {} if solver_kwargs is None else solver_kwargs

    # ---------- SDP constraints -------
    # W: the (unnormalised, up to tr(W)=1) fully decomposable witness.
    W: cp.Variable = cp.Variable((d, d), hermitian=True)
    constraints: list[Constraint] = [cp.real(cp.trace(W)) == 1]

    for mask in bipartitions:
        # P_M >= 0 is one half of decomposability w.r.t. bipartition M.
        P_M: cp.Variable = cp.Variable((d, d), hermitian=True)
        constraints.append(P_M >> 0)

        # The other half: Q_M = (W - P_M)^{T_M} >= 0.
        Q_M = compute_pt_cvxpy(dim, W - P_M, mask)
        constraints.append(Q_M >> 0)

    # ---------- SDP objective ----------
    # Tr(rho W) is real for Hermitian rho, W.
    # cp.real() guards against imaginary parts introduced by the solver's (real, symmetric) representation of complex Hermitian variables.
    objective = cp.Minimize(cp.real(cp.trace(rho @ W)))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose, **solver_kwargs)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solver did not converge (status: {problem.status}).")

    minval = float(np.real(np.trace(rho @ W.value)))

    if minval < -precision:
        witness = np.asarray(W.value)
    else:
        witness = np.zeros((d, d), dtype=complex)

    return DecomposableWitnessResult(minval=minval, witness=witness)

def compute_decomposable_monotone(
    dim: Sequence[int],
    rho: np.ndarray,
    solver: Optional[str] = None,
    precision: float = 1e-12,
    verbose: bool = False,
    solver_kwargs: Optional[dict] = None,
) -> float:
    """
    Compute the genuine multipartite entanglement monotone genuine multipartite negativity.

    Parameters
    ----------
    dim : sequence of intpr
        Local Hilbert space dimension of each of the ``N`` subsystems, with ``d = prod(dim)``.
    rho : np.ndarray
        ``(d, d)`` array describing the state.
    solver : str, optional
        Name of the CVXPY-registered conic solver to use (e.g. ``"SCS"``, ``"CVXOPT"``, ``"MOSEK"``).
        If ``None``, CVXPY chooses an installed solver automatically (typically SCS).
    precision : float, optional
        Numerical threshold below which the optimal value is considered strictly negative (i.e. entanglement detected).
    verbose : bool, optional
        If True, print solver progress output.
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to ``cvxpy.Problem.solve`` (e.g. ``{"eps": 1e-8}`` to tighten SCS's convergence tolerance).

    Returns
    -------
    float
        The value of the entanglement monotone for the input state.

    Raises
    ------
    ValueError
        If ``rho`` is not a valid quantum state, or if ``dim`` is inconsistent with ``rho``.
    RuntimeError
        If the SDP solver fails to reach an optimal (or optimal inaccurate) solution.
    """
    # ---------- Validations ----------
    # Hilbert space dimension
    d = int(np.prod(dim))
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("`rho` must be a square matrix.")
    if rho.shape != (d, d):
        raise ValueError("`dim` is incompatible with `rho`.")
    if np.min(dim) < 2:
        raise ValueError("All local dimensions must be at least 2.")
    
    # Density matrix
    is_valid, _ = compute_is_valid_dm(rho=rho, tol=precision)
    if not is_valid:
        raise ValueError(f"`rho` is not Hermitian within tolerance {precision} ")
    
    # ---------- Definitions ----------
    # Compute all possible bipartitions of {1, ..., N}
    N = len(dim)
    bipartitions = enumerate_bipartitions(N)

    # Define arguments for the solver
    solver_kwargs = {} if solver_kwargs is None else solver_kwargs

    identity = np.eye(d)

    # ---------- SDP constraints -------
    # W: the (unnormalised, up to tr(W)=1) fully decomposable witness.
    W: cp.Variable = cp.Variable((d, d), hermitian=True)
    constraints: list[Constraint] = []

    for mask in bipartitions:
        # 0 <= P_M <= I is one half of decomposability w.r.t. bipartition M.
        P_M: cp.Variable = cp.Variable((d, d), hermitian=True)
        constraints.append(P_M >> 0)
        constraints.append(identity - P_M >> 0)

        # The other half: 0 <= Q_M = (W - P_M)^{T_M} <= I.
        Q_M = compute_pt_cvxpy(dim, W - P_M, mask)
        constraints.append(Q_M >> 0)
        constraints.append(identity - Q_M >> 0)

   # ---------- SDP objective ----------
    # Tr(rho W) is real for Hermitian rho, W.
    # cp.real() guards against imaginary parts introduced by the solver's (real, symmetric) representation of complex Hermitian variables.
    objective = cp.Minimize(cp.real(cp.trace(rho @ W)))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose, **solver_kwargs)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solver did not converge (status: {problem.status}).")

    minval = float(np.real(np.trace(rho @ W.value)))
    return -minval