#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np

# Auxiliary python functions
from typing import Tuple, Dict, Union, Optional

# ----------------------------------------
# State preparation and validation
# ----------------------------------------

def compute_is_valid_dm(rho: np.ndarray, tol: float = 1e-10, return_info: bool = False) -> Tuple[bool, Dict]:
    """
    Check whether a matrix is a valid density matrix.

    A valid density matrix must:
    - Have unit trace
    - Be Hermitian
    - Be positive semi-definite (PSD)

    Parameters
    ----------
    rho : ndarray
        Input matrix of shape (d, d).
    tol : float, optional
        Numerical tolerance for all checks (default is 1e-10).
    return_info : bool, optional
        If True, also return a dictionary with diagnostic information.
    
    Returns
    -------
    bool or (bool, dict)
        - True if rho is a valid density matrix within tolerance.
        - If `return_info=True`, returns (is_valid, diagnostics_dict).
    
    Raises
    ------
    TypeError
        If rho is not a numpy array.
    ValueError
        If rho is not a square matrix.
    """
    # Input validations
    if not isinstance(rho, np.ndarray):
        raise TypeError("'rho' must be a numpy array.")
    
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("'rho' must be a square matrix.")
    
    d = rho.shape[0]
    
    # Trace check
    tr = np.trace(rho).real
    trace_ok = abs(tr - 1.0) <= tol
    
    hermitian_ok = np.allclose(rho, rho.conj().T, atol=tol)
    
    # Hermiticity and positivity checks
    if hermitian_ok:
        eigvals = np.linalg.eigvalsh(rho)
        min_eig = eigvals[0]
        psd_ok = min_eig >= -tol
    else:
        eigvals = None
        min_eig = None
        psd_ok = False
    
    # Return results
    is_valid = trace_ok and hermitian_ok and psd_ok
    if not return_info:
        return is_valid
    
    # Diagnostic information
    info = {
        "trace": tr,
        "trace_ok": trace_ok,
        "hermitian_ok": hermitian_ok,
        "psd_ok": psd_ok,
        "min_eigenvalue": min_eig,
    }

    return is_valid, info

def generate_rand_dm(d: int, rank: int) -> np.ndarray:
    """
    Generate a random density matrix of given dimension and rank using the Ginibre ensemble.

    Parameters
    ----------
    d : int
        Dimension of the Hilbert space. Must be a positive integer.
    rank : int
        Desired rank of the density matrix. Must satisfy 1 <= rank <= d.
    rng : numpy.random.Generator, optional
        Random number generator instance. If None, a new default generator is created.
    
    Returns
    -------
    ndarray
        A complex ndarray of shape (d, d) representing a density matrix with trace 1 and rank equal to 'rank'.
    
    Raises
    ------
    TypeError
        If inputs are not of the expected type.
    ValueError
        If `d <= 0`, `rank <= 0`, or `rank > d`.
    """
    # Input validations
    if not isinstance(d, int):
        raise TypeError("Input 'd' must be an integer.")
    if not isinstance(rank, int):
        raise TypeError("Input 'rank' must be an integer.")
    
    if d <= 0:
        raise ValueError("Input 'd' must be a positive integer.")
    if rank <= 0:
        raise ValueError("Input 'rank' must be a positive integer.")
    if rank > d:
        raise ValueError("Input 'rank' must be less than or equal to 'd'.")
    
    # Random generator initialization
    rng = np.random.default_rng()
    
    # Compute density matrix
    X = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    rho = X @ X.conj().T
    tr_val = np.trace(rho).real
    # Trace validation
    if tr_val <= 0:
        raise RuntimeError("Generated matrix has non-positive trace.")
    
    return rho / tr_val

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

def compute_binary_entropy(x: Union[np.ndarray, float], tol: float = 1e-12) -> Union[np.ndarray, float]:
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

def compute_eof(rho: Optional[np.ndarray] = None, C: Optional[float] = None) -> float:
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