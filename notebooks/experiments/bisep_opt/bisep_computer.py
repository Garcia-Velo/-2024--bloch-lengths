import numpy as np


def _proj(psi):
    """Projector |psi><psi| for a pure state given as a 1D (or 1D-like) array."""
    psi = np.asarray(psi, dtype=complex).ravel()
    return np.outer(psi, psi.conj())


def _block(qs, psi_single, psi_pair):
    """sum_n q_n * |psi_single_n><psi_single_n| (x) |psi_pair_n><psi_pair_n|"""
    return sum(q * np.kron(_proj(s), _proj(p))
               for q, s, p in zip(qs, psi_single, psi_pair))


def _permute_systems(rho, dims, perm):
    """
    Reorder the tensor-factor structure of a density matrix.

    `rho` currently lives on subsystems ordered according to `perm`:
    its k-th tensor factor corresponds to natural (target) index perm[k].
    `dims` gives the local dimensions in the natural/target order (A,B,C,...).
    """
    n = len(dims)
    if list(perm) == list(range(n)):
        return rho
    cur_dims = [dims[p] for p in perm]
    inv = np.argsort(perm)                       # where each natural index currently sits
    T = rho.reshape(cur_dims + cur_dims)
    T = T.transpose(list(inv) + list(inv + n))
    d = int(np.prod(dims))
    return T.reshape(d, d)


def biseparable_state(p,
                       psiA, qA, psiBC,
                       psiB, qB, psiAC,
                       psiC, qC, psiAB):
    """
    Build the three-party biseparable state

        rho = p_A * rho_{A|BC} + p_B * rho_{B|AC} + p_C * rho_{C|AB}

    with  rho_{A|BC} = sum_i qA_i |psiA_i><psiA_i| (x) |psiBC_i><psiBC_i|
    (and analogously for the B|AC and C|AB terms).

    Parameters
    ----------
    p : (pA, pB, pC), sums to 1
    psiA, psiB, psiC   : sequences of local pure-state vectors (one per term)
    psiBC, psiAC, psiAB: sequences of pure-state vectors on the pair Hilbert
                          spaces H_B⊗H_C, H_A⊗H_C, H_A⊗H_B respectively
    qA, qB, qC : probability weights for each mixture, each sums to 1

    Returns
    -------
    rho : ndarray of shape (dA*dB*dC, dA*dB*dC), density matrix on H_A⊗H_B⊗H_C
    """
    pA, pB, pC = p

    dA = np.asarray(psiA[0]).size
    dB = np.asarray(psiB[0]).size
    dC = np.asarray(psiC[0]).size
    dims = [dA, dB, dC]

    rho_A = _block(qA, psiA, psiBC)                                            # already (A,B,C)
    rho_B = _permute_systems(_block(qB, psiB, psiAC), dims, perm=[1, 0, 2])     # (B,A,C) -> (A,B,C)
    rho_C = _permute_systems(_block(qC, psiC, psiAB), dims, perm=[2, 0, 1])     # (C,A,B) -> (A,B,C)

    return pA * rho_A + pB * rho_B + pC * rho_C


if __name__ == "__main__":
    # --- quick sanity check with three qubits ---
    zero, one = np.array([1, 0]), np.array([0, 1])
    plus = (zero + one) / np.sqrt(2)
    bell = (np.kron(zero, zero) + np.kron(one, one)) / np.sqrt(2)  # on the pair space

    p = [0.5, 0.3, 0.2]

    psiA, qA, psiBC = [zero, one], [0.6, 0.4], [bell, np.kron(zero, one)]
    psiB, qB, psiAC = [plus], [1.0], [bell]
    psiC, qC, psiAB = [zero, plus], [0.7, 0.3], [np.kron(zero, one), bell]

    rho = biseparable_state(p, psiA, qA, psiBC, psiB, qB, psiAC, psiC, qC, psiAB)

    print("shape:", rho.shape)
    print("trace:", np.trace(rho).real)
    print("Hermitian:", np.allclose(rho, rho.conj().T))
    eigs = np.linalg.eigvalsh(rho)
    print("min eigenvalue:", eigs.min(), " (should be >= 0)")
