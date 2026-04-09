#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np
from scipy.optimize import minimize

# Auxiliary python functions
from typing import Tuple, Dict

#------------------------------
# Parametrization of the density matrix
#------------------------------

def compute_param_from_X(X: np.ndarray) -> np.ndarray:
    """
    Convert a complex matrix into a real parameter vector.

    Parameters
    ----------
    X : ndarray
        Complex array of shape (d, d).
    
    Returns
    -------
    ndarray
        Real vector of length 2 * d^2 containing real and imaginary parts.
    
    Raises
    ------
    ValueError
        If input is not a 2D complex array.
    """
    # Validate input: must be a 2D numpy array
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("'X' must be a 2D numpy array.")
    
    # Flatten the matrix into a 1D array in row-major order
    vals = X.ravel(order='C')
    
    # Create an empty array to hold real and imaginary parts
    x = np.empty(2 * vals.size, dtype=float)
    
    # Assign real parts to even indices and imaginary parts to odd indices
    x[0::2] = vals.real
    x[1::2] = vals.imag

    return x

def compute_X_from_param(x: np.ndarray) -> np.ndarray:
    
    """
    Construct a matrix X from a real parameter vector x.

    Parameters
    ----------
    x : ndarray
        Real vector of length 2 * d^2 representing a complex matrix.
    
    Returns
    -------
    X : ndarray
        Complex matrix encoding a quantum state.
    
    Raises
    ------
    ValueError
        If input size is invalid.
    """
    # Validate input: must be a 1D numpy array
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("'x' must be a 1D numpy array.")
    
    # Check if length is even (real and imaginary parts)
    if len(x) % 2 != 0:
        raise ValueError("Length of 'x' must be even.")
    
    # Calculate number of complex elements and etermine dimension
    n_complex = len(x) // 2
    d = int(np.sqrt(n_complex))
    
    # Validate that n_complex is a perfect square
    if d * d != n_complex:
        raise ValueError("Invalid size: cannot reshape into square matrix.")
    
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
                            R_target: Dict[Tuple[int, ...], float], tol: float = 1e-12) -> Tuple[float, np.ndarray]:
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
        X = compute_X_from_param(x)
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
    scalar = np.trace(grad_rho @ rho).real
    grad_X = (2 / trXX) * (grad_rho @ X - scalar * X)
    grad = np.empty_like(x)
    grad_vals = grad_X.ravel()
    grad[0::2] = grad_vals.real
    grad[1::2] = grad_vals.imag

    return loss, grad

def compute_initial_param(d, tensor_basis: np.ndarray, subset_index_map: Dict[Tuple[int, ...], np.ndarray], Rt: Dict[Tuple[int, ...], float]) -> np.ndarray:
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
    
    Returns
    -------
    ndarray
        Optimized real parameter vector x.
    """
    # Initialize random number generator
    rng = np.random.default_rng()
    
    # Initialize variables to track the best optimization result
    best_loss = np.inf
    best_res = None
    
    # Try multiple random initializations to find the best starting point
    for _ in range(5):
        
        # Generate random initial parameter vector
        x0 = rng.normal(size=(2*d**2))
        
        try:
            # Perform optimization using L-BFGS-B method
            res = minimize(initial_state_loss_grad, x0,
                           args=(tensor_basis, subset_index_map, Rt),
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
    x = res.x
    return x