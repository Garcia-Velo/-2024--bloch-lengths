#------------------------------
# Importations
#------------------------------

# Numerical and scientific python programming
import numpy as np

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
    # Perform input validations.
    if not isinstance(rho, np.ndarray):
        raise TypeError("'rho' must be a numpy array.")
    
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("'rho' must be a square matrix.")
    
    d = rho.shape[0]
    
    # Trace check.
    tr = np.trace(rho).real
    trace_ok = abs(tr - 1.0) <= tol
    
    # Hermitian check.
    hermitian_ok = np.allclose(rho, rho.conj().T, atol=tol)
    
    # Positivity checks
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