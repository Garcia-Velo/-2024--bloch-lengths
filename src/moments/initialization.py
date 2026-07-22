#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np
from scipy.optimize import minimize, OptimizeResult

# Auxiliary python functions
from dataclasses import dataclass
from typing import Any

# Local importations
from moments.bloch import compute_bloch_vector, compute_bloch_norms_from_vector
from moments.quantum import compute_is_valid_dm

# ----------------------------------------
# Definitions
# ----------------------------------------

@dataclass
class ParameterResult:
    """
    Data class to store the results of the moment-preserving entanglement optimization.

    Attributes
    ----------
    param : np.ndarray
        The optimized parameters.
    rho : np.ndarray
        The optimized density matrix.
    bloch : Dict[Tuple[int, ...], np.ndarray]
        Final Bloch vectors for each subsystem subset.
    moments : Dict[Tuple[int, ...], float]
        Final moment norms for each subsystem subset.
    loss : float
        Final value of the loss.
    checks : Dict[str, Any]
        Additional checks and information about the optimization.
    optimizer_info : Dict[str, Any]
        Information about the optimization process.
    """
    param: np.ndarray
    rho: np.ndarray
    bloch: dict[tuple[int, ...], np.ndarray]
    moments: dict[tuple[int, ...], float]
    loss: float
    checks: dict[str, Any]
    optimizer_info: dict[str, Any]

#------------------------------
# Parametrization of the density matrix
#------------------------------

def compute_param_from_X(X: np.ndarray, cholesky: bool = False) -> np.ndarray:
    """
    Convert a complex matrix into a real parameter vector.

    Parameters
    ----------
    X : ndarray
        Complex array of shape (d, d).
    cholesky : bool, default=False
        If True, assumes X is lower triangular and only stores non-zero elements.
    
    Returns
    -------
    ndarray
        Real vector containing real and imaginary parts of the matrix elements.
        If cholesky=False: length 2 * d^2
        If cholesky=True: length d*(d+1) (only lower triangular elements)
    
    Raises
    ------
    ValueError
        If input is not a 2D complex array, or if cholesky=True but matrix is not lower triangular.
    """
    # Validate input: must be a 2D numpy array
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("'X' must be a 2D numpy array.")
    
    d = X.shape[0]
    if X.shape[1] != d:
        raise ValueError("'X' must be a square matrix.")
    
    # For the cholesky parametrization, we only store the lower triangular elements.
    if cholesky:

        # Check if matrix is lower triangular
        if not np.allclose(np.triu(X, k=1), 0):
            raise ValueError("With cholesky=True, X must be lower triangular.")
        
        # Indices for lower triangular part
        lower_indices = np.tril_indices(d)
        vals = X[lower_indices]
        
        # Create parameter vector
        x = np.empty(2 * len(vals), dtype=float)
        x[0::2] = np.real(vals)
        x[1::2] = np.imag(vals)
        
        return x
    
    # For the general case, we store all elements of the matrix.
    else:

        # Flatten the matrix into a 1D array in row-major order
        vals = X.ravel(order='C')

        # Create an empty array to hold real and imaginary parts
        x = np.empty(2 * vals.size, dtype=float)

        # Assign real parts to even indices and imaginary parts to odd indices
        x[0::2] = np.real(vals)
        x[1::2] = np.imag(vals)

        return x

def compute_X_from_param(x: np.ndarray, cholesky: bool = False) -> np.ndarray:
    """
    Construct a matrix X from a real parameter vector x.

    Parameters
    ----------
    x : ndarray
        Real vector representing a complex matrix.
    cholesky : bool, default=False
        If True, reconstructs a lower triangular matrix from a compact parameter vector.
    
    Returns
    -------
    X : ndarray
        Complex matrix.
    
    Raises
    ------
    ValueError
        If input size is invalid or inconsistent with cholesky parameter.
    """
    # Validate input: must be a 1D numpy array
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("'x' must be a 1D numpy array.")
    
    # Check if length is even (real and imaginary parts)
    if len(x) % 2 != 0:
        raise ValueError("Length of 'x' must be even.")
    
    # Calculate number of complex elements
    n_complex = len(x) // 2
    
    # For the cholesky parametrization, we only recover the lower triangular elements.
    if cholesky:

        # Determine output dimension
        discriminant = 1 + 8 * n_complex
        d = int((-1 + np.sqrt(discriminant)) / 2)
        
        # Verify that n_complex matches triangular number
        if d * (d + 1) // 2 != n_complex:
            raise ValueError(f"Invalid size: for cholesky=True, n_complex={n_complex} "
                           f"must be a triangular number d*(d+1)/2 for some integer d.")
        
        # Reconstruct complex values from real and imaginary parts
        vals = x[0::2] + 1j * x[1::2]
        
        # Create empty matrix and fill lower triangular part
        X = np.zeros((d, d), dtype=complex)
        lower_indices = np.tril_indices(d)
        X[lower_indices] = vals
    
    # For the general case, we recover all elements of the matrix.
    else:

        # Determine output dimension
        d = int(np.sqrt(n_complex))
        
        # Validate that n_complex is a perfect square
        if d * d != n_complex:
            raise ValueError("Invalid size: cannot reshape into square matrix. ")
        
        # Reconstruct complex values from real and imaginary parts
        vals = x[0::2] + 1j * x[1::2]
        
        # Reshape into square matrix
        X = vals.reshape((d, d))
        
    return X

def compute_dm_from_X(X: np.ndarray) -> np.ndarray:
    """
    Compute the density matrix from a complex matrix X.

    Parameters
    ----------
    X : ndarray
        Complex matrix of shape (d, d).
    
    Returns
    -------
    ndarray
        Density matrix rho = X @ X.conj().T / Tr(X @ X.conj().T).
    
    Raises
    ------
    RuntimeError
        If the constructed matrix has non-positive trace.
    """
    # Compute the unnormalized density matrix as X X†
    rho = X @ X.conj().T
    
    # Calculate the trace of the unnormalized density matrix
    trace_val = np.trace(rho).real
    
    # Check if the trace is positive to ensure validity
    if trace_val <= 0:
        raise RuntimeError("Constructed matrix has non-positive trace.")
    
    # Normalize the density matrix by dividing by its trace
    rho /= trace_val
    
    return rho

#------------------------------
# Auxiliary functions for optimization
#------------------------------

def build_parameter_result(tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray], Rt: dict[tuple[int, ...], float],
                           optimizer_res: OptimizeResult, cholesky: bool = False, psd_tol: float = 1e-10) -> ParameterResult:
    """
    Build the ParameterResult from optimized parameters and optimizer result.

    Parameters
    ----------
    x : np.ndarray
        Optimized parameter vector.
    optimizer_res : OptimizeResult
        Result from scipy.optimize.minimize.
    d : int
        Dimension of the system.
    tensor_basis : ndarray
        Tensor-product operator basis, shape (K, D, D).
    subset_index_map : dict
        Mapping from subsystem subsets to indices in tensor_basis.
    Rt : dict
        Target Bloch vector norms for each subsystem subset.
    cholesky : bool, default=False
        If True, uses compact Cholesky parametrization (lower triangular X).
    psd_tol : float, default=1e-10
        Tolerance for positive semidefinite check.
    
    Returns
    -------
    ParameterResult
        Complete result with all computed quantities.
    """
    # Extract final state
    x = optimizer_res.x
    X = compute_X_from_param(x, cholesky)
    rho = compute_dm_from_X(X)
    r = compute_bloch_vector(tensor_basis, subset_index_map, rho)
    R = compute_bloch_norms_from_vector(r)
    
    # Run checks on the solution
    moments_distance = {}
    for subset in subset_index_map.keys():
        moments_distance[subset] = float(abs(R[subset] - Rt[subset]))
    
    is_valid_dm, is_valid_dm_info = compute_is_valid_dm(rho, psd_tol)
    
    checks = {
        "moments_distance": moments_distance,
        "is_valid_dm": is_valid_dm,
        "is_valid_dm_info": is_valid_dm_info,
    }

    # Returns information about optimization
    optimizer_info = {
        "mode": "moment_preserving_bloch",
        "success": bool(optimizer_res.success),
        "message": str(optimizer_res.message),
        "nit": int(optimizer_res.nit),
        "nfev": int(optimizer_res.nfev),
    }

    return ParameterResult(
        param=x,
        rho=rho,
        bloch=r,
        moments=R,
        loss=float(optimizer_res.fun),
        checks=checks,
        optimizer_info=optimizer_info,
    )

#------------------------------
# Optimization functions
#------------------------------

def initial_state_loss_grad(x: np.ndarray, tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray],
                            R_target: dict[tuple[int, ...], float], cholesky: bool = False, tol: float = 1e-12) -> tuple[float | np.floating[Any], np.ndarray]:
    """
    Objective function and its gradient for optimization of a density matrix
    with prescribed Bloch vector norms.
    
    Parameters
    ----------
    x : ndarray
        Real parameter vector representing a complex matrix X.
    R_target : dict
        Target Bloch vector norms for each subsystem subset.
    tensor_basis : ndarray
        Tensor-product operator basis, shape (K, D, D).
    subset_index_map : dict
        Mapping from subsystem subsets to indices in tensor_basis.
    cholesky : bool, default=False
        If True, assumes X is lower triangular and uses compact parametrization.
    tol : float
        Tolerance for norm to avoid division by zero.
    
    Returns
    -------
    loss : float
        Sum of squared deviations between Bloch norms and targets.
    grad : ndarray
        Gradient of the loss w.r.t. x.
    """
    # Compute the density matrix from X
    try:
        X = compute_X_from_param(x, cholesky=cholesky)
        rho = compute_dm_from_X(X)
    except Exception as e:
        raise ValueError(f"Error reconstructing density matrix: {e}")
    
    # Compute Bloch vector components
    r = np.einsum('aij,ji->a', tensor_basis, rho, optimize=True).real
    
    # Initialize gradient
    grad_rho = np.zeros_like(rho, dtype=complex)
    
    # Loop over each subsystem and compute squared loss
    loss = 0.0
    for subset, indices in subset_index_map.items():
        r_M = r[indices]
        norm = np.linalg.norm(r_M)
        diff = norm - R_target[subset]
        loss += diff**2
        if norm > tol:
            # Normalize the direction vector
            direction = r_M / norm
            
            # Add to gradient using tensor product
            grad_rho += 2 * diff * np.tensordot(direction, tensor_basis[indices], axes=1)
    
    # Compute density matrix
    XX = X @ X.conj().T

    trXX = np.trace(XX).real
    if trXX <= 0:
        raise RuntimeError("Trace of X X† is non-positive.")
    
    # Compute gradient
    scalar = np.real(np.trace(grad_rho @ rho))
    grad_X = (2 / trXX) * (grad_rho @ X - scalar * X)
    # Convert gradient to parameter space based on parametrization
    if cholesky:
        # Only keep lower triangular part (consistent with parametrization)
        lower_indices = np.tril_indices(grad_X.shape[0])
        grad_vals = grad_X[lower_indices]
        grad = np.empty(2 * len(grad_vals), dtype=float)
        
    else:
        grad = np.empty_like(x)
        grad_vals = grad_X.ravel()
    
    grad[0::2] = np.real(grad_vals)
    grad[1::2] = np.imag(grad_vals)

    return loss, grad

def compute_initial_param(d: int, tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray], Rt: dict[tuple[int, ...], float],
                          cholesky: bool = False, psd_tol: float = 1e-10) -> OptimizeResult:
    """
    Compute initial parameters for the density matrix by optimizing over random initializations.

    Parameters
    ----------
    d : int
        Dimension of the system.
    tensor_basis : ndarray
        Tensor-product operator basis, shape (K, D, D).
    subset_index_map : dict
        Mapping from subsystem subsets to indices in tensor_basis.
    Rt : dict
        Target Bloch vector norms for each subsystem subset.
    cholesky : bool, default=False
        If True, uses compact Cholesky parametrization (lower triangular X).
    psd_tol : float, default=1e-10
        Tolerance for positive semidefinite check.
    return_full : bool, default=True
        If True, return full ParameterResult. If False, return the optimizer result.
    
    Returns
    -------
    OptimizeResult
    """
    # Determine number of parameters based on parametrization
    if cholesky:
        n_params = d * (d + 1)  # Number of real parameters for lower triangular
    else:
        n_params = 2 * d * d     # Number of real parameters for full matrix
    
    # Generate random initial parameter vector
    rng = np.random.default_rng()
    x0 = rng.normal(size=n_params)

    # Perform optimization using L-BFGS-B method
    res = minimize(initial_state_loss_grad, x0,
                           args=(tensor_basis, subset_index_map, Rt, cholesky),
                           jac=True, method='L-BFGS-B')
    
    return res

def compute_initial_param_repeat(d: int, tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray], Rt: dict[tuple[int, ...], float],
                                 cholesky: bool = False, psd_tol: float = 1e-10, attempts: int = 5) -> ParameterResult:
    """
    Compute initial parameters by trying multiple random initializations and selecting the best result.

    Parameters
    ----------
    d : int
        Dimension of the system.
    tensor_basis : ndarray
        Tensor-product operator basis, shape (K, D, D).
    subset_index_map : dict
        Mapping from subsystem subsets to indices in tensor_basis.
    Rt : dict
        Target Bloch vector norms for each subsystem subset.
    cholesky : bool, default=False
        If True, uses compact Cholesky parametrization (lower triangular X).
    psd_tol : float, default=1e-10
        Tolerance for positive semidefinite check.
    attempts : int, default=5
        Number of random initializations to try.
    
    Returns
    -------
    ParameterResult
        The best result among the attempts.
    """
    # Initialize variables to track the best optimization result
    loss_best = np.inf
    res_best = None

    # Try multiple random initializations to find the best starting point
    for _ in range(attempts):
        
        try:
            # Perform optimization
            res = compute_initial_param(d, tensor_basis, subset_index_map, Rt, cholesky, psd_tol)
        
        except Exception as e:
            # Print error message if minimization fails
            print(f"Minimization failed at point R = {tuple(Rt.values())}: {e}")
            continue
        
        # Update best result if current loss is lower
        if res.fun < loss_best:
            loss_best = res.fun
            res_best = res
    
    if res_best is None:
        raise RuntimeError(f"All {attempts} optimization attempts failed at R = {tuple(Rt.values())}.")
    
    # Build the full result for the best optimizer result
    return build_parameter_result(tensor_basis, subset_index_map, Rt, res_best, cholesky, psd_tol)