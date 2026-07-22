#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np

# Auxiliary python functions
from functools import reduce
from typing import Sequence

# ----------------------------------------
# Basic algebra
# ----------------------------------------

def compute_adj(A: np.ndarray) -> np.ndarray:
    """
    Compute the adjoint (Hermitian conjugate) of a matrix.

    Parameters
    ----------
    A : np.ndarray
        Input matrix or vector.

    Returns
    -------
    np.ndarray
        The adjoint of ``A`` with the same shape transposed.

    Raises
    ------
    ValueError
        If ``A`` is not a NumPy array or is empty.
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("A must be a numpy array.")
    if A.size == 0:
        raise ValueError("A must not be an empty array.")
    return np.conj(A).T

def compute_tensor_product(O_v: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute the tensor (Kronecker) product of a sequence of operators.

    Parameters
    ----------
    O_v : Sequence[np.ndarray]
        Non-empty sequence of NumPy arrays representing operators or
        vectors.

    Returns
    -------
    np.ndarray
        Tensor product of all arrays in ``O_v``.

    Raises
    ------
    ValueError
        If ``O_v`` is not a non-empty list or tuple, if any element is not a NumPy array, if any array is empty, or if the tensor product cannot be computed due to incompatible shapes.
    """
    if not isinstance(O_v, (list, tuple)) or len(O_v) == 0:
        raise ValueError("O_v must be a non-empty list or tuple of numpy arrays.")
    for idx, o in enumerate(O_v):
        if not isinstance(o, np.ndarray):
            raise ValueError(f"Element at index {idx} is not a numpy array.")
        if o.size == 0:
            raise ValueError(f"Element at index {idx} is an empty array.")
    try:
        return reduce(np.kron, O_v)
    except ValueError as e:
        raise ValueError("Some arrays in O_v have incompatible shapes for Kronecker product.") from e

def compute_outer_product(psi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the outer product of two state vectors.

    Parameters
    ----------
    psi : np.ndarray
        Left state vector of shape ``(d, 1)``.
    phi : np.ndarray
        Right state vector of shape ``(d, 1)``.

    Returns
    -------
    np.ndarray
        Matrix representing the outer product.

    Raises
    ------
    ValueError
        If ``psi`` and ``phi`` do not have the same shape or are not
        column vectors.
    """
    if psi.shape != phi.shape:
        raise ValueError("psi and phi must have the same shape.")
    if len(psi.shape) != 2 or psi.shape[1] != 1:
        raise ValueError("psi and phi must be column vectors of shape (d, 1).")
    return psi @ compute_adj(phi)

# ----------------------------------------
# State preparation and validation
# ----------------------------------------

def compute_is_valid_dm(rho: np.ndarray, tol: float = 1e-10) -> tuple[bool, dict]:
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

def generate_GHZ_state(N: int) -> np.ndarray:
    """
    Generate the N-qubit GHZ state
    .. math::

        |\\mathrm{GHZ}_N\\rangle =
        \\frac{1}{\\sqrt{2}(|0\\rangle^{\\otimes N} + |1\\rangle^{\\otimes N}).
    
    Parameters
    ----------
    N : int
        Number of qubits.
    
    Returns
    -------
    np.ndarray
        Complex column vector of shape ``(2**N, 1)`` containing the GHZ state.
        If ``N <= 0``, a zero column vector of shape ``(2**max(0, N), 1)`` is returned.
    """
    if N <= 0:
        return np.zeros((2**max(0, N), 1), dtype=complex)
    GHZ = np.zeros((2**N, 1), dtype=complex)
    GHZ[0, 0] = 1
    GHZ[2**N - 1, 0] = 1
    return GHZ / np.sqrt(2)

def generate_W_state(N: int) -> np.ndarray:
    """
    Generate the N-qubit W state
    .. math::

        |W_N\\rangle =
        \\frac{1}{\\sqrt{N}} (|0\\cdots01\\rangle + |1\\cdots00\\rangle).
    
    Parameters
    ----------
    N : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        Complex column vector of shape ``(2**N, 1)`` containing the W state.
        If ``N <= 0``, a zero column vector of shape ``(2**max(0, N), 1)`` is returned.
    """
    if N <= 0:
        return np.zeros((2**max(0, N), 1), dtype=complex)
    W = np.zeros((2**N, 1), dtype=complex)
    for n in range(N):
        W[2**n, 0] = 1
    return W / np.sqrt(N)