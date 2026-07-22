#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np

from scipy.linalg import eigh

# Auxiliary python functions
from typing import Sequence, Iterable

# ----------------------------------------
# Entanglement measures
# ----------------------------------------

_YY = np.kron(1j * np.array([[0, -1], [1, 0]], dtype = complex), 1j * np.array([[0, -1], [1, 0]], dtype = complex))

def compute_concurrence(rho: np.ndarray) -> float:
    """
    Compute the Wootters concurrence of a two-qubit density matrix.
    Parameters
    ----------
    YY : ndarray
        Precomputed Pauli-Y ⊗ Pauli-Y matrix of shape (4,4).
    rho : ndarray
        Two-qubit density matrix of shape (4,4).
    
    Returns
    -------
    float
        Concurrence value in [0,1].
    
    Raises
    ------
    ValueError
        If 'rho' is not 4x4 or 'YY' is not 4x4.
    """
    # Input validations
    if rho.shape != (4, 4):
        raise ValueError("'rho' must be a 4x4 matrix.")
    if _YY.shape != (4, 4):
        raise ValueError("'YY' must be a 4x4 matrix.")
    
    # Compoute auxiliary matrix
    R = rho @ _YY @ rho.conj() @ _YY
    
    # Compute the concurrance using the eigenvalues of R
    eigvals = np.linalg.eigvals(R)
    eigvals = np.clip(np.sqrt(np.maximum(eigvals, 0)), 0, None)
    eigvals.sort()
    eigvals = eigvals[::-1]

    return max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]).real

def compute_binary_entropy(x: np.ndarray | float, tol: float = 1e-12) -> np.ndarray | float:
    """
    Compute the binary entropy function H(x) = -x log2 x - (1-x) log2 (1-x),
    with tolerance for limiting cases.

    Parameters
    ----------
    x : float
        Probability value (expected in [0,1]).
    tol : float, optional
        Tolerance to treat values very close to 0 or 1 as exact, by default 1e-12.

    Returns
    -------
    float
        Binary entropy. Returns 0 if x is within tolerance of 0 or 1.

    Notes
    -----
    This avoids numerical issues when x is extremely close to 0 or 1.
    """
    # Treat scalars and arrays uniformly
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float64)
    
    # Only compute entropy where valid (avoids log2(0) warnings)
    mask = (x > tol) & (x < 1.0 - tol)
    if mask.any():
        x_valid = x[mask]
        result[mask] = -x_valid * np.log2(x_valid) - (1 - x_valid) * np.log2(1 - x_valid)
    
    # Return a float if input was scalar, otherwise return array
    return result.item() if result.ndim == 0 else result

def compute_eof(rho: np.ndarray | None = None, C: np.ndarray | float | None = None) -> np.ndarray | float:
    """
    Compute the entanglement of formation for a two-qubit state.

    Parameters
    ----------
    rho : ndarray, optional
        Two-qubit density matrix of shape (4,4). Required if C is None.
    C : float, optional
        Concurrence value. Required if rho is None.

    Returns
    -------
    float
        Entanglement of formation (in bits).

    Raises
    ------
    ValueError
        If neither 'rho' nor 'C' is provided.
    """
    # Input validations
    if C is None:
        if rho is None:
            raise ValueError("Pass either the density matrix or the concurrence value.")
        # Conpute EoF from density matrix
        C = compute_concurrence(rho)
    
    # Conpute EoF from concurrence
    sqrt_term = np.sqrt(1 - C**2)
    p = 0.5 + 0.5 * sqrt_term
    
    return compute_binary_entropy(p)

def compute_tr_norm(eigenvalues: np.ndarray | None = None, A: np.ndarray | None = None) -> float:
    """
    Compute the trace norm of a Hermitian matrix.
    For a Hermitian matrix A, the trace norm reduces to the sum of absolute eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray, optional
        Precomputed eigenvalues of A. If provided, A is not used.
    A : np.ndarray, optional
        A Hermitian matrix of shape (d, d). Required if eigenvalues is not provided.
    
    Returns
    -------
    float
        The trace norm of A.
    """
    if eigenvalues is None:
        if A is None:
            raise ValueError("Provide eigenvalues or matrix.")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        if not np.allclose(A, A.conj().T):
            raise ValueError("Matrix must be Hermitian.")
        eigvals = eigh(A, eigvals_only=True)
    else:
        eigvals = eigenvalues.copy()
    
    return float(np.sum(np.abs(eigvals)))

def compute_pt(dim: Sequence[int], A: np.ndarray, subsystem: int | Iterable[int] = 1) -> np.ndarray:
    """
    Compute the partial transpose of a multipartite operator.

    Parameters
    ----------
    dim : Sequence[int]
        Local Hilbert-space dimensions.
    A : ndarray
        Matrix of shape (prod(dim), prod(dim)).
    subsystem : int or iterable[int], default=1
        Which subsystem(s) to transpose.

    Returns
    -------
    ndarray
        Partial transpose.
    """
    dim = tuple(dim)
    N = len(dim)
    d = int(np.prod(dim))

    if A.shape != (d, d):
        raise ValueError("A must have shape (prod(dim), prod(dim)).")

    if isinstance(subsystem, int):
        subsystem = (subsystem,)
    else:
        subsystem = tuple(subsystem)

    if any(s < 0 or s >= N for s in subsystem):
        raise ValueError("Invalid subsystem index.")

    # Reshape into a tensor with one input and one output index per subsystem.
    T = A.reshape(dim + dim)

    # Swap input/output indices for the selected subsystems.
    perm = list(range(2 * N))
    for s in subsystem:
        perm[s], perm[N + s] = perm[N + s], perm[s]

    return T.transpose(perm).reshape(d, d)

def compute_pt_norm(dim: list[int], eigenvalues: np.ndarray | None = None, A: np.ndarray | None = None, subsystem: int = 0) -> float:
    """
    Compute the trace norm  of the the partial transpose of a bipartite density matrix.

    Parameters
    ----------
    dim : List[int]
        Local Hilbert space dimensions.
    eigenvalues : np.ndarray, optional
        Precomputed eigenvalues of rho^Gamma. If provided, rho is not used.
    A : np.ndarray, optional
        Matrix to be partially transposed. Required if eigenvalues is not provided.
    subsystem : int
        Subsystem to partially transpose: 0 for A, 1 for B (default).

    Returns
    -------
    float
        The trace norm of rho^Gamma.
    
    Raises
    ------
    ValueError
        If 'rho' is not a square matrix or if 'dim' is inconsistent with the shape of 'rho'.
    """
    if eigenvalues is not None:
        return compute_tr_norm(eigenvalues=eigenvalues)
    elif A is None:
        raise ValueError("Must provide either A or eigenvalues.")
    else:
        d = A.shape[0]
        if dim[0] * dim[1] != d:
            raise ValueError(f"dim={dim} inconsistent with A shape ({d}, {d}).")
        
        A_pt = compute_pt(dim=dim, A=A, subsystem=subsystem)
        return compute_tr_norm(A=A_pt)

def compute_pt_norm_jac(dim: list[int], eigenvalues: np.ndarray | None = None, eigenvectors: np.ndarray | None = None,
                        A: np.ndarray | None = None, subsystem: int = 0) -> np.ndarray:
    """
    Compute the Jacobian of the trace norm of the partial transpose with respect to the original matrix A.

    Parameters
    ----------
    dim : List[int]
        Local Hilbert space dimensions.
    eigenvalues : np.ndarray, optional
        Precomputed eigenvalues of rho^Gamma. If provided, rho is not used. If provided, U must also be provided.
    eigenvectors : np.ndarray, optional
        Unitary matrix of eigenvectors of rho^Gamma. Required if eigenvalues is provided.
    A : np.ndarray, optional
        Matrix to be partially transposed. Required if eigenvalues is not provided.
    subsystem : int
        Subsystem to partially transpose: 0 for A, 1 for B (default).

    Returns
    -------
    np.ndarray
        The Jacobian of the trace norm of the partial transpose with respect to A.
    """
    if eigenvalues is not None and eigenvectors is not None:
        eigvals = eigenvalues.copy()
        eigvecs = eigenvectors.copy()
    elif A is None:
        raise ValueError("Must provide either A or both eigenvalues and eigenvectors.")
    else:
        eigvals, eigvecs = eigh(A)
    S = (eigvecs * np.sign(eigvals)) @ eigvecs.conj().T
    return compute_pt(dim, S)

def compute_negativity(
    dim: Sequence[int] | None = None,
    rho: np.ndarray | None = None,
    trace_norm_pt: float | np.ndarray | None = None,
    subsystem: int | Sequence[int] = 1,
    ) -> float | np.ndarray:
    """
    Compute the entanglement negativity of a bipartite quantum state.

    Accepts either the full density matrix or a precomputed trace norm of the partial transpose.

    Parameters
    ----------
    rho : np.ndarray, optional
        Density matrix of shape (dA*dB, dA*dB). Required if trace_norm_pt is not provided.
    trace_norm_pt : float, optional
        Precomputed trace norm ||rho^Gamma||_1. If provided, rho and dim are not used.
    dim : List[int], optional
        Local Hilbert space dimensions (dA, dB). Required if rho is provided and the subsystems are not equal in dimension.
        If None and rho is provided, a symmetric bipartition dA = dB = sqrt(n) is assumed.
    subsystem : int
        Subsystem to partially transpose: 0 for A, 1 for B (default).

    Returns
    -------
    float
        The entanglement negativity N(rho) >= 0.

    Raises
    ------
    ValueError
        If neither rho nor trace_norm_pt is provided, or if dim is inconsistent with the shape of rho.
    """
    if trace_norm_pt is not None:
        return (trace_norm_pt - 1.0) / 2.0

    if rho is None:
        raise ValueError("Must provide either rho or trace_norm_pt.")

    n = rho.shape[0]
    if dim is None:
        dA = dB = int(np.sqrt(n))
        if dA * dB != n:
            raise ValueError(
                "Could not infer a symmetric bipartition from rho. "
                "Please provide dim=(dA, dB) explicitly."
            )
        dim = [dA, dB]
    elif np.prod(dim) != n:
        raise ValueError(f"dim={dim} inconsistent with rho shape.")

    rho_pt = compute_pt(dim, rho, subsystem)
    return (compute_tr_norm(A=rho_pt) - 1.0) / 2.0