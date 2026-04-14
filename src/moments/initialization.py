#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np
from scipy.optimize import minimize

# Auxiliary python functions
from typing import Tuple, Dict, Union, Any

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
# Optimization functions
#------------------------------

def initial_state_loss_grad(x: np.ndarray, tensor_basis: np.ndarray, subset_index_map: Dict[Tuple[int, ...], np.ndarray],
                            R_target: Dict[Tuple[int, ...], float], cholesky: bool = False, tol: float = 1e-12) -> Tuple[Union[float, np.floating[Any]], np.ndarray]:
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

def optmize_initial_param(d: int, tensor_basis: np.ndarray, subset_index_map: Dict[Tuple[int, ...], np.ndarray],
                          Rt: Dict[Tuple[int, ...], float], cholesky: bool = False) -> np.ndarray:
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
    
    Returns
    -------
    ndarray
        Optimized real parameter vector x.
    """
    # Initialize random number generator
    rng = np.random.default_rng()
    
    # Determine number of parameters based on parametrization
    if cholesky:
        n_params = d * (d + 1)  # Number of real parameters for lower triangular
    else:
        n_params = 2 * d * d     # Number of real parameters for full matrix
    
    # Initialize variables to track the best optimization result
    best_loss = np.inf
    best_res = None
    
    # Try multiple random initializations to find the best starting point
    for _ in range(5):
        
        # Generate random initial parameter vector
        x0 = rng.normal(size=n_params)
        
        try:
            # Perform optimization using L-BFGS-B method
            res = minimize(initial_state_loss_grad, x0,
                           args=(tensor_basis, subset_index_map, Rt, cholesky),
                           jac=True, method='L-BFGS-B')
        except Exception as e:
            # Print error message if minimization fails
            print(f"Minimization failed at point R = {tuple(Rt.values())}: {e}")
            continue
        
        # Update best result if current loss is lower
        if res.fun < best_loss:
            best_loss = res.fun
            best_res = res
    
    # Extract the optimized parameters from the best result
    if best_res is not None:
        return best_res.x
    else:
        raise RuntimeError("Optimization failed for all initializations.")