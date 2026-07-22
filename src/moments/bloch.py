#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np

# Auxiliary python functions
from itertools import product, combinations, chain
from typing import Sequence, Any

#------------------------------
# Definitions
#------------------------------

def generate_pauli_basis() -> np.ndarray:
    """
    Initialize the Pauli basis of a one-qubit system. This includes the identity matrix as the first element of the basis.

    Returns
    -------
    ndarray
        Array of shape (4, 2, 2), where the indices 0, 1, 2 and 3 correspond to the I, X, Y and Z matrices respectively.
    """
    # Definition of the identity and Paui matrices
    sigma_0 = np.identity(2, dtype = complex)
    sigma_1 = np.array([[0, 1], [1, 0]], dtype = complex)
    sigma_2 = 1j * np.array([[0, -1], [1, 0]], dtype = complex)
    sigma_3 = np.array([[1, 0], [0, -1]], dtype = complex)

    # Return as a numpy asrray for easier handling
    return np.array([sigma_0, sigma_1, sigma_2, sigma_3])

import numpy as np


def generate_gell_mann_basis(d: int, normalize: bool = True) -> np.ndarray:
    """
    Generate the generalized Gell-Mann basis for a qudit of local dimension d.
    The basis consists of d^2 Hermitian matrices, including the identity as the first element.

    The matrices satisfy:
        - Lambda_0 = I_d (identity)
        - Tr(Lambda_i) = 0 for i > 0 (traceless)
        - Tr(Lambda_i @ Lambda_j) = d * delta_ij (orthogonal, not normalized)
        - Lambda_i = Lambda_i^dagger (Hermitian)

    Parameters
    ----------
    d : int
        Local Hilbert space dimension. Must satisfy d >= 2.
    
    Returns
    -------
    np.ndarray
        Array of shape (d², d, d) of dtype complex128.
        Index 0 is always the identity (normalized to I/sqrt(d) if normalized=True).
        Remaining d²-1 entries are the traceless generators, ordered as:
            - symmetric off-diagonal (d(d-1)/2 matrices)
            - antisymmetric off-diagonal (d(d-1)/2 matrices)
            - diagonal (d-1 matrices)
    Raises
    ------
    ValueError
        If d < 2.
    """
    if d < 2:
        raise ValueError(f"Dimension must be d >= 2, got {d}.")
    
    # Normalization scale
    if normalize:
        scale = np.sqrt(d / 2)
    else:
        scale = 1

    basis = []

    # Identity
    basis.append(np.eye(d, dtype=complex))

    # Symmetric off-diagonal: |j><k| + |k><j| for j > k ---
    for k in range(d):
        for j in range(k):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = 1.0
            m[k, j] = 1.0
            basis.append(scale * m)
    
    # --- Antisymmetric off-diagonal: -i|j><k| + i|k><j| for j < k ---
    for k in range(d):
        for j in range(k):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = -1j
            m[k, j] =  1j
            basis.append(scale * m)
    
    # --- Diagonal: sqrt(2/l(l+1)) * (sum_{j<l} |j><j| - l|l><l|) ---
    for l in range(1, d):
        m = np.zeros((d, d), dtype=complex)
        diag_norm = np.sqrt(2.0 / (l * (l + 1)))
        for j in range(l):
            m[j, j] = 1.0
        m[l, l] = -l
        prefactor = np.sqrt(2.0 / (l * (l + 1)))
        basis.append(scale * diag_norm * m)
    
    return np.array(basis)

#------------------------------
# Auxiliary functions
#------------------------------

def compute_bipartite_region_upper(d: int, a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """
    Compute upper bound of the region of valid bipartite states of local dimension d.
    For more information, see Theorem 1 in 10.1103/PhysRevA.109.012423.

    Parameters
    ----------
    d : int
        Local Hilbert space dimension (must be >= 2).
    a : float or np.ndarray
        Euclidean norm ||a|| of the local Bloch vector of the first subsystem.
        Must be non-negative.
    b : float or np.ndarray
        Euclidean norm ||b|| of the local Bloch vector of the second subsystem.
        Must be non-negative.

    Returns
    -------
    t : float or np.ndarray
        The computed value of ||t||. Same shape as broadcasted inputs.

    Raises
    ------
    ValueError
        If d < 2.
        If a or b contain negative values.
        If the argument inside the square root becomes negative
        (i.e., invalid physical region)..
    """

    # Convert inputs to arrays for vectorization
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Checks
    if d < 2:
        raise ValueError("Dimension d must be >= 2.")

    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Norms ||a|| and ||b|| must be non-negative.")

    delta = np.abs(a - b)

    # Compute argument inside the square root
    radicand = d**2 - 1 - a**2 - b**2 - d * np.sqrt(2 * d) * delta + d * delta**2

    if np.any(radicand < 0):
        raise ValueError(
            "Negative value inside square root encountered. "
            "Inputs are outside the physically allowed region."
        )

    return np.sqrt(radicand)

def compute_bipartite_region_ent(d: int, a: float | np.ndarray, b: float | np.ndarray, tol: float = 1e-12) -> float | np.ndarray:
    """
    Compute lower bound of the region of entangled bipartite states of local dimension d.
    For more information, see Observation 8 in 10.1103/PhysRevLett.126.150501.

    Parameters
    ----------
    d : int
        Local Hilbert space dimension (must be >= 2).
    a : float or np.ndarray
        Euclidean norm ||a|| of the local Bloch vector of the first subsystem.
        Must be non-negative.
    b : float or np.ndarray
        Euclidean norm ||b|| of the local Bloch vector of the second subsystem.
        Must be non-negative.

    Returns
    -------
    t : float or np.ndarray
        Computed value of ||t||. Shape follows NumPy broadcasting rules.

    Raises
    ------
    ValueError
        If d < 2.
        If any of a or b are negative.
        If result is negative (non-physical region detected).
    """

    a2 = np.asarray(a, dtype=float)**2
    b2 = np.asarray(b, dtype=float)**2

    # Validity checks
    if d < 2:
        raise ValueError("d must be >= 2.")

    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Norms ||a|| and ||b|| must be non-negative.")

    # Compute the two branches
    term1 = (d - 1) * a2 - b2
    term2 = (d - 1) * b2 - a2

    min_term = np.minimum(term1, term2)

    t2 = (d - 1) + min_term

    # Physical consistency check
    if np.any(t2 < - tol):
        raise ValueError(
            "Computed ||t||^2 is negative. Inputs likely outside physical region."
        )

    return np.sqrt(t2 + tol)

def compute_bipartite_region_lower(dim: list[int], a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """
    Compute lower bound of the region of valid bipartite states of local dimension d.
    For more information, see Lemma 7 in 10.1016/j.laa.2019.09.008.

    Parameters
    ----------
    dim : List[int]
        Local Hilbert space dimensions (dA, dB) (all must be >= 2).
    a : float or np.ndarray
        Euclidean norm ||a|| of the local Bloch vector of the first subsystem.
        Must be non-negative.
    b : float or np.ndarray
        Euclidean norm ||b|| of the local Bloch vector of the second subsystem.
        Must be non-negative.

    Returns
    -------
    t : float or np.ndarray
        Computed value of ||t||. Shape follows NumPy broadcasting rules.

    Raises
    ------
    ValueError
        If dA < 2 or dB < 2.
        If any of a or b are negative.
        If computed ||t|| is negative (outside physical region).
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    dA, dB = dim

    # Validity checks
    if dA < 2 or dB < 2:
        raise ValueError("dA and dB must both be >= 2.")

    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Norms ||a|| and ||b|| must be non-negative.")

    # Precompute square roots
    alpha = np.sqrt(dA - 1)
    beta = np.sqrt(dB - 1)

    # Compute t norm
    t = beta * a + alpha * b - alpha * beta
    bound = np.maximum(0, t)

    # Physical consistency check
    if np.any(bound < 0):
        raise ValueError(
            "Computed ||t|| is negative. Inputs may lie outside the physical region."
        )

    return bound

#------------------------------
# Main functions
#------------------------------

def compute_tensor_basis(local_bases: Sequence[np.ndarray]) -> np.ndarray:
    """
    Construct the tensor-product operator basis for a composite quantum system.
    
    Parameters
    ----------
    local_bases : sequence of ndarray
        List of local operator bases. Each element must be an array of shape (k_n, d_n, d_n), where:
        - k_n is the number of basis operators for subsystem n,
        - d_n is the local Hilbert space dimension.
    
    Returns
    -------
    ndarray
        Array of shape (k, d, d), where:
        - k = product of all k_n,
        - D = product of all d_n.
        Each entry is a tensor-product basis operator.
    
    Raises
    ------
    ValueError
        If input shapes are inconsistent.
    """
    # Validate that local_bases is a non-empty sequence
    if not isinstance(local_bases, Sequence) or len(local_bases) == 0:
        raise ValueError("'local_bases' must be a non-empty sequence.")
    
    # Initialize lists to store local dimensions and basis sizes
    dims = []
    sizes = []

    # Extract dimensions and sizes from each local basis
    for B in local_bases:
        # Verify each basis is a 3D array
        if not isinstance(B, np.ndarray) or B.ndim != 3:
            raise ValueError("Each local basis must be an array of shape (k_n, d_n, d_n).")
        k, d1, d2 = B.shape
        # Verify matrices are square
        if d1 != d2:
            raise ValueError("Local basis matrices must be square.")
        # Store the Hilbert space dimension and number of basis operators
        dims.append(d1)
        sizes.append(k)
    
    # Compute total Hilbert space dimension as product of local dimensions
    D = int(np.prod(dims))
    # Compute total number of basis operators as product of local sizes
    K = int(np.prod(sizes))

    # Allocate array to store all tensor-product basis operators
    basis = np.empty((K, D, D), dtype=complex)

    # Generate all combinations of basis operator indices across subsystems
    for idx, multi_idx in enumerate(product(*[range(k) for k in sizes])):
        # Start with the first local basis operator from subsystem 0
        op = local_bases[0][multi_idx[0]]
        # Iteratively tensor-product with operators from remaining subsystems
        for n in range(1, len(local_bases)):
            op = np.kron(op, local_bases[n][multi_idx[n]])
        # Store the resulting tensor-product operator
        basis[idx] = op
    
    return basis

def compute_subset_index_map(local_basis_sizes: Sequence[int]) -> dict[tuple[int, ...], np.ndarray]:
    """
    Compute index mappings for all subsystem subsets in a tensor-product basis.

    Parameters
    ----------
    local_basis_sizes : sequence of int
        Number of basis elements for each subsystem (e.g., 4 for qubits, d_n^2 for qudits with Gell-Mann basis).
    
    Returns
    -------
    dict
        Dictionary mapping subsets (tuples of subsystem indices starting at 0)
        to arrays of indices in the full tensor basis.
    
    Raises
    ------
    ValueError
        If inputs are invalid.
    
    Notes
    -----
    The identity element (index 0) is excluded for each subsystem.
    """
    # Validate that all local basis sizes are positive integers
    if not all(isinstance(k, int) and k > 0 for k in local_basis_sizes):
        raise ValueError("All local basis sizes must be positive integers.")
    
    # Get the number of subsystems
    N = len(local_basis_sizes)

    # Initialize dictionary to store subset-to-indices mappings
    subset_index_map = {}
    
    # Compute stride values for converting multi-indices to flat indices in tensor product basis
    powers = np.cumprod([1] + list(local_basis_sizes[::-1]))[:-1][::-1]
    
    # Generate all non-empty subsets of subsystems (1-indexed for compatibility)
    subsystems = range(1, N + 1)
    subsets = chain.from_iterable(
        combinations(subsystems, r) for r in range(1, N + 1)
    )
    
    # Process each subset of subsystems
    for subset in subsets:
        subset = tuple(subset)
        # Convert subsystem indices from 1-indexed to 0-indexed
        subset_shift = tuple(i - 1 for i in subset)
        # Get the basis sizes for this particular subset
        sizes = [local_basis_sizes[i] for i in subset_shift]
        
        # Create multi-dimensional grids for all combinations of basis indices (excluding identity at index 0)
        grids = np.meshgrid(*[np.arange(1, k) for k in sizes], indexing='ij')
        # Stack grids and reshape to 2D array where each row is a multi-index
        grid = np.stack(grids, axis=-1).reshape(-1, len(subset))

        # Get the stride values corresponding to this subset
        weights = powers[list(subset_shift)]
        # Compute flat indices by matrix-vector multiplication (dot product with strides)
        indices = grid @ weights

        # Store the computed indices for this subset
        subset_index_map[subset] = indices.astype(int)
    
    return subset_index_map

def compute_bloch_vector(tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray], rho: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    """
    Compute the generalized Bloch vector coefficients of a density matrix.

    Parameters
    ----------
    tensor_basis : ndarray
        Array of shape (k, d, d) containing the operator basis.
    subset_index_map : dict
        Mapping from subsystem subsets to basis indices.
    rho : ndarray
        Density matrix of shape (d, d).
    
    Returns
    -------
    dict
        Dictionary mapping each subset to its Bloch vector components.
    
    Raises
    ------
    ValueError
        If dimensions are inconsistent.
    """
    # Dimension validation and extraction
    if tensor_basis.ndim != 3:
        raise ValueError("'tensor_basis' must have shape (k, d, d).")
    
    k, d1, d2 = tensor_basis.shape

    if rho.shape != (d1, d2):
        raise ValueError("Shape mismatch between 'rho' and 'tensor_basis'.")
    
    # Compute Bloch coefficients
    coeffs = np.einsum('aij,ji->a', tensor_basis, rho, optimize=True)

    # Arrange Bloch coefficiens into local Bloch vectors
    return {subset: coeffs[indices].real
            for subset, indices in subset_index_map.items()}

def compute_dm_from_bloch(tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray],
                  bloch_coeffs: dict[tuple[int, ...], np.ndarray], identity_index: int = 0) -> np.ndarray:
    """
    Reconstruct a density matrix from generalized Bloch coefficients.

    Parameters
    ----------
    tensor_basis : ndarray
        Operator basis of shape (k, d, d).
    subset_index_map : dict
        Mapping from subsets to basis indices.
    bloch_coeffs : dict
        Dictionary of Bloch vectors for each subset.
    identity_index : int, optional
        Index of the identity operator in the basis (default is 0).
    
    Returns
    -------
    ndarray
        Density matrix of shape (d, d).
    
    Raises
    ------
    ValueError
        If inputs are inconsistent.
    
    Notes
    -----
    Assumes the operator basis is orthonormal under the Hilbert-Schmidt inner product: Tr(B_i^† B_j) = δ_ij.
    The density matrix is reconstructed as: rho = sum_i c_i B_i, with c_identity fixed by normalization Tr(rho) = 1.
    """
    # Dimension validation and extraction
    if tensor_basis.ndim != 3:
        raise ValueError("'tensor_basis' must have shape (k, d, d).")
    
    k, d, _ = tensor_basis.shape

    # Initialize density matrix
    rho_coeffs = np.zeros(k, dtype=float)

    # Arrange local Bloch vectors into the global Bloch vector
    for subset, indices in subset_index_map.items():
        if subset not in bloch_coeffs:
            raise ValueError(f"Missing Bloch coefficients for subset {subset}.")
        rho_coeffs[indices] = bloch_coeffs[subset]
    
    # Add identity term
    rho_coeffs[identity_index] = 1.0
    
    # Compute density matrix from Bloch coefficients
    rho = np.einsum('a,aij->ij', rho_coeffs, tensor_basis, optimize=True)

    # Normalization
    return (1/d) * rho

def compute_bloch_norms_from_dm(tensor_basis: np.ndarray, subset_index_map: dict[tuple[int, ...], np.ndarray],
                        rho: np.ndarray) -> dict[tuple[int, ...], np.floating[Any]]:
    """
    Compute the norms of Bloch vector components for all subsystem subsets.

    Parameters
    ----------
    tensor_basis : ndarray
        Operator basis of shape (k, d, d).
    subset_index_map : dict
        Mapping from subsets to basis indices.
    rho : ndarray
        Density matrix of shape (d, d).
    
    Returns
    -------
    dict
        Dictionary mapping each subset to the Euclidean norm of its Bloch vector.
    
    Raises
    ------
    ValueError
        If dimensions are inconsistent.
    """
    # Dimension validation and extraction
    if tensor_basis.ndim != 3:
        raise ValueError("'tensor_basis' must have shape (k, d, d).")
    
    k, d1, d2 = tensor_basis.shape

    if rho.shape != (d1, d2):
        raise ValueError("Shape mismatch between 'rho' and 'tensor_basis'.")
    
    # Compute Bloch coefficients
    coeffs = np.einsum('aij,ji->a', tensor_basis, rho, optimize=True).real

    # Arrange Bloch coefficiens into local Bloch vectors and compute norms
    return {subset: np.linalg.norm(coeffs[indices])
            for subset, indices in subset_index_map.items()}

def compute_bloch_norms_from_vector(bloch_coeffs: dict[tuple[int, ...], np.ndarray], tol: float = 0.0) -> dict[tuple[int, ...], float]:
    """
    Compute the Euclidean norms of Bloch vectors for all subsystem subsets.

    Parameters
    ----------
    bloch_coeffs : dict
        Dictionary mapping subsets (tuples) to Bloch vectors (ndarrays).
    tol : float, optional
        Threshold below which norms are set to zero (default is 0.0).
    
    Returns
    -------
    dict
        Dictionary mapping each subset to the norm of its Bloch vector.
    
    Raises
    ------
    TypeError
        If input is not a dictionary or values are not numpy arrays.
    ValueError
        If any Bloch vector is not 1D.
    """
    # Validate datatypes
    if not isinstance(bloch_coeffs, dict):
        raise TypeError("'bloch_coeffs' must be a dictionary.")
    
    result = {}
    for subset, vec in bloch_coeffs.items():
        if not isinstance(vec, np.ndarray):
            raise TypeError(f"Bloch vector for subset {subset} must be a numpy array.")
        if vec.ndim != 1:
            raise ValueError(f"Bloch vector for subset {subset} must be 1D.")
        
        # Compute Bloch norms
        n = np.linalg.norm(vec)
        if n <= tol:
            n = 0.0
        result[subset] = n
    
    return result