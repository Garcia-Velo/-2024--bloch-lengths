#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np
import cvxpy as cp

from scipy import sparse

# Auxiliary python functions
from typing import Sequence
from numpy.typing import NDArray
from scipy.sparse import sparray, spmatrix

# ----------------------------------------
# Utilities for decomposable witness
# ----------------------------------------

def enumerate_bipartitions(N: int) -> list[NDArray[np.int_]]:
    """
    Enumerate all inequivalent bipartitions of ``N`` subsystems.

    A bipartition ``M | complement(M)`` is equivalent to ``complement(M) | M``, and neither ``M`` nor its complement may be empty.
    The most significant bit (i.e. the mask entry for subsystem 0) is always 0; such that subsystem 0 by convenction always lies in the complement of ``M``.

    Parameters
    ----------
    N : int
        Total number of subsystems, ``N >= 2``.

    Returns
    -------
    list of np.ndarray
        A list of ``2**(N - 1) - 1`` binary mask vectors (each of length ``N``, dtype int), one per inequivalent, non-trivial bipartition.

    Raises
    ------
    ValueError
        If ``N < 2``.
    """
    if N < 2:
        raise ValueError("At least two subsystems are required to form a bipartition.")

    masks = []
    for m in range(1, 2 ** (N - 1)):
        bits = format(m, f"0{N}b")
        masks.append(np.array([int(b) for b in bits], dtype=int))
    return masks

# ----------------------------------------
# Partial transpose
# ----------------------------------------

def _swap_axes_for_mask(dim: Sequence[int], operator: np.ndarray, mask: NDArray[np.int_]) -> np.ndarray:
    """
    Core reshape-and-swap-axes routine (no input validation).

    Parameters
    ----------
    dim : sequence of int
        Local dimension of every subsystem.
    operator : np.ndarray
        The ``d``-by-``d`` matrix to transform (any dtype).
    mask : sequence of int
        Binary indicator (0 or 1) selecting which subsystems are transposed.
    
    Returns
    -------
    np.ndarray
        The transformed ``d``-by-``d`` matrix.
    """
    N = len(dim)
    dimension = int(np.prod(dim))
    tensor = operator.reshape(tuple(dim) + tuple(dim))

    axes = list(range(2 * N))
    for k in range(N):
        if mask[k]:
            # Swap the "row" axis k with its "column" counterpart n + k.
            axes[k], axes[N + k] = axes[N + k], axes[k]

    transposed_tensor = np.transpose(tensor, axes)
    # np.transpose returns a view; make it contiguous before the final reshape.
    return np.ascontiguousarray(transposed_tensor).reshape(dimension, dimension)

def _compute_pt_permutation(dim: Sequence[int], mask: NDArray[np.int_]) -> sparray | spmatrix:
    """
    Build the sparse permutation matrix representing the partial transpose.

    Parameters
    ----------
    dim : sequence of int
        Local dimension of every subsystem.
    mask : sequence of int
        Binary indicator selecting which subsystems are transposed.

    Returns
    -------
    scipy.sparse.csr_matrix
        A ``D**2``-by-``D**2`` permutation matrix (``D = prod(dims)``).
    """
    d = int(np.prod(dim))

    # row-major flat index of entry (i, j).
    flat_index = np.arange(d * d, dtype=np.int64).reshape(d, d)
    # Apply the *same* reshape/swap-axes logic to the index grid
    source_index = _swap_axes_for_mask(dim, flat_index, mask).astype(np.int64)

    destination = np.arange(d * d)
    source = source_index.reshape(-1)  # row-major flatten, matches `destination` ordering
    data = np.ones(d * d)

    return sparse.coo_matrix(
        (data, (destination, source)), shape=(d * d, d * d)
        ).tocsr()

def compute_pt_cvxpy(dim: Sequence[int], operator_expr: "cp.Expression", mask: NDArray[np.int_]) -> "cp.Expression":
    """
    Partial transpose of a CVXPY affine expression, for use in SDP constraints.

    CVXPY does not natively support permuting axes of a tensor reshaping
    of a matrix expression, so the partial transpose is instead applied
    as multiplication by a constant (real, sparse) permutation matrix on
    the row-major-vectorised operator -- see
    :func:`_partial_transpose_permutation`. This yields an expression
    that CVXPY treats as affine in ``operator_expr``, exactly as needed
    to write constraints such as ``partial_transpose_cvxpy(W - P, M) >> 0``.

    Parameters
    ----------
    dim : sequence of int
        Local dimension of every subsystem.
    operator_expr : cvxpy.Expression
        A ``(d, d)`` CVXPY expression (e.g. ``W - P`` where ``W`` and ``P`` are Hermitian CVXPY variables), with ``d = prod(dims)``.
    mask : sequence of int
        Binary indicator selecting which subsystems are transposed.

    Returns
    -------
    cvxpy.Expression
        A ``(d, d)`` CVXPY expression representing the partial transpose of ``operator_expr``.
    """
    d = int(np.prod(dim))

    permutation_matrix = _compute_pt_permutation(dim, mask)

    vectorised = cp.reshape(operator_expr, (d * d,), order="C")
    permuted_vector = permutation_matrix @ vectorised
    return cp.reshape(permuted_vector, (d, d), order="C")