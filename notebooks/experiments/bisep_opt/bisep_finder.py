import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')

class ClosestBiSeparableFinder:
    """
    Finds the closest bi-separable state to a given three-qubit state
    using optimization over convex combinations of product states.
    """
    
    def __init__(self, rho_target, decomposition_length=4, method='SLSQP'):
        self.rho_target = rho_target
        self.d = 8
        self.n = decomposition_length
        
        assert rho_target.shape == (self.d, self.d), "Target state must be 8x8"
        assert np.allclose(rho_target, rho_target.conj().T), "Target must be Hermitian"
        assert np.isclose(np.trace(rho_target), 1.0), "Trace must be 1"
        
        self.partitions = [
            {'name': 'A|BC', 'subsystem_dims': (2, 4)},
            {'name': 'B|AC', 'subsystem_dims': (2, 4)},
            {'name': 'C|AB', 'subsystem_dims': (2, 4)}
        ]
        
        self.method = method
        
    def random_pure_state(self, dim):
        """Generate a random pure state of given dimension."""
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        return psi / np.linalg.norm(psi)
    
    def pure_state_to_dm(self, psi):
        """Convert pure state vector to density matrix."""
        return np.outer(psi, psi.conj())
    
    def tensor_product_dm(self, dm1, dm2):
        """Compute tensor product of two density matrices."""
        return np.kron(dm1, dm2)
    
    def create_product_state(self, params, partition_idx):
        """
        Create a product state density matrix for a given bipartition.
        Returns both the density matrix and the decomposition information.
        """
        dims = self.partitions[partition_idx]['subsystem_dims']
        d1, d2 = dims
        
        if partition_idx == 0:  # A|BC
            n_params_A = 2 * d1
            psi_A_vec = params[:n_params_A]
            psi_A = psi_A_vec[:d1] + 1j * psi_A_vec[d1:]
            psi_A = psi_A / np.linalg.norm(psi_A)
            rho_A = self.pure_state_to_dm(psi_A)
            
            psi_BC_vec = params[n_params_A:]
            psi_BC = psi_BC_vec[:d2] + 1j * psi_BC_vec[d2:]
            psi_BC = psi_BC / np.linalg.norm(psi_BC)
            rho_BC = self.pure_state_to_dm(psi_BC)
            
            rho = self.tensor_product_dm(rho_A, rho_BC)
            decomposition = {
                'partition': 'A|BC',
                'state_A': psi_A,
                'state_BC': psi_BC,
                'rho_A': rho_A,
                'rho_BC': rho_BC
            }
            
        elif partition_idx == 1:  # B|AC
            n_params_B = 2 * d1
            psi_B_vec = params[:n_params_B]
            psi_B = psi_B_vec[:d1] + 1j * psi_B_vec[d1:]
            psi_B = psi_B / np.linalg.norm(psi_B)
            rho_B = self.pure_state_to_dm(psi_B)
            
            psi_AC_vec = params[n_params_B:]
            psi_AC = psi_AC_vec[:d2] + 1j * psi_AC_vec[d2:]
            psi_AC = psi_AC / np.linalg.norm(psi_AC)
            rho_AC = self.pure_state_to_dm(psi_AC)
            
            rho_BA_C = self.tensor_product_dm(rho_B, rho_AC)
            rho = self.permute_to_standard(rho_BA_C, partition_idx)
            decomposition = {
                'partition': 'B|AC',
                'state_B': psi_B,
                'state_AC': psi_AC,
                'rho_B': rho_B,
                'rho_AC': rho_AC
            }
            
        else:  # C|AB
            n_params_C = 2 * d1
            psi_C_vec = params[:n_params_C]
            psi_C = psi_C_vec[:d1] + 1j * psi_C_vec[d1:]
            psi_C = psi_C / np.linalg.norm(psi_C)
            rho_C = self.pure_state_to_dm(psi_C)
            
            psi_AB_vec = params[n_params_C:]
            psi_AB = psi_AB_vec[:d2] + 1j * psi_AB_vec[d2:]
            psi_AB = psi_AB / np.linalg.norm(psi_AB)
            rho_AB = self.pure_state_to_dm(psi_AB)
            
            rho_CA_B = self.tensor_product_dm(rho_C, rho_AB)
            rho = self.permute_to_standard(rho_CA_B, partition_idx)
            decomposition = {
                'partition': 'C|AB',
                'state_C': psi_C,
                'state_AB': psi_AB,
                'rho_C': rho_C,
                'rho_AB': rho_AB
            }
        
        return rho, decomposition
    
    def permute_to_standard(self, rho, partition_idx):
        """Permute density matrix to standard qubit ordering (A,B,C)."""
        if partition_idx == 0:
            return rho
        
        if partition_idx == 1:  # B|AC: |b,a,c> -> |a,b,c>
            perm = np.zeros((8, 8))
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        old_idx = (b << 2) | (a << 1) | c
                        new_idx = (a << 2) | (b << 1) | c
                        perm[new_idx, old_idx] = 1
            return perm @ rho @ perm.T
            
        else:  # C|AB: |c,a,b> -> |a,b,c>
            perm = np.zeros((8, 8))
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        old_idx = (c << 2) | (a << 1) | b
                        new_idx = (a << 2) | (b << 1) | c
                        perm[new_idx, old_idx] = 1
            return perm @ rho @ perm.T
    
    def create_bi_separable_state_with_decomposition(self, all_params):
        """
        Create a bi-separable state and return its full decomposition.
        """
        idx = 0
        
        # Extract mixture weights for the three partitions
        mixture_weights = all_params[idx:idx+3]
        idx += 3
        mixture_weights = self.softmax(mixture_weights)
        
        d1, d2 = self.partitions[0]['subsystem_dims']
        params_per_state = 2 * d1 + 2 * d2
        
        rho_bi_sep = np.zeros((self.d, self.d), dtype=complex)
        
        # Store the full decomposition
        decomposition = {
            'mixture_weights': mixture_weights,
            'partitions': []
        }
        
        # For each bipartition
        for part_idx in range(3):
            part_decomposition = {
                'name': self.partitions[part_idx]['name'],
                'weight': mixture_weights[part_idx],
                'terms': []
            }
            
            # Extract weights for the n terms
            part_weights = all_params[idx:idx+self.n]
            idx += self.n
            part_weights = self.softmax(part_weights)
            
            rho_partition = np.zeros((self.d, self.d), dtype=complex)
            
            # Create the convex combination
            for k in range(self.n):
                state_params = all_params[idx:idx+params_per_state]
                idx += params_per_state
                rho_product, state_decomp = self.create_product_state(state_params, part_idx)
                
                rho_partition += part_weights[k] * rho_product
                
                part_decomposition['terms'].append({
                    'weight': part_weights[k],
                    'state_info': state_decomp
                })
            
            rho_bi_sep += mixture_weights[part_idx] * rho_partition
            decomposition['partitions'].append(part_decomposition)
        
        return rho_bi_sep, decomposition
    
    def softmax(self, x):
        """Softmax function to ensure weights are positive and sum to 1."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def objective_function(self, params):
        """Compute the distance between target and bi-separable state."""
        rho_bi_sep, _ = self.create_bi_separable_state_with_decomposition(params)
        diff = self.rho_target - rho_bi_sep
        return np.real(np.trace(diff @ diff.conj().T))
    
    def initialize_params(self):
        """Initialize random parameters for optimization."""
        d1, d2 = self.partitions[0]['subsystem_dims']
        params_per_state = 2 * d1 + 2 * d2
        
        total_params = 3 + 3 * self.n + 3 * self.n * params_per_state
        params = np.random.randn(total_params) * 0.1
        return params
    
    def optimize(self, n_restarts=5, max_iter=1000):
        """Run optimization with multiple random restarts."""
        best_result = None
        best_distance = np.inf
        all_results = []
        
        for restart in range(n_restarts):
            print(f"Restart {restart + 1}/{n_restarts}")
            
            x0 = self.initialize_params()
            
            if self.method == 'differential_evolution':
                bounds = [(-5, 5)] * len(x0)
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    maxiter=max_iter,
                    popsize=15,
                    tol=1e-8,
                    disp=False
                )
            else:
                options = {
                    'maxiter': max_iter,
                    'ftol': 1e-10,
                    'disp': False
                }
                result = minimize(
                    self.objective_function,
                    x0,
                    method=self.method,
                    options=options
                )
            
            distance = result.fun
            all_results.append({
                'distance': distance,
                'success': result.success,
                'params': result.x
            })
            
            print(f"  Distance: {distance:.6e}, Success: {result.success}")
            
            if distance < best_distance:
                best_distance = distance
                best_result = result
        
        # Get the final decomposition
        rho_bi_sep, decomposition = self.create_bi_separable_state_with_decomposition(best_result.x)
        
        return {
            'rho_bi_separable': rho_bi_sep,
            'distance': best_distance,
            'success': best_result.success,
            'decomposition': decomposition,  # Added decomposition
            'all_results': all_results
        }


def print_decomposition(decomposition):
    """
    Pretty print the bi-separable decomposition.
    """
    print("\n" + "="*60)
    print("BI-SEPARABLE DECOMPOSITION")
    print("="*60)
    
    mixture_weights = decomposition['mixture_weights']
    print(f"\nPartition mixture weights:")
    for i, part in enumerate(decomposition['partitions']):
        print(f"  {part['name']}: {mixture_weights[i]:.4f}")
    
    print(f"\nSum of partition weights: {np.sum(mixture_weights):.6f}")
    
    for part in decomposition['partitions']:
        print(f"\n{'-'*40}")
        print(f"Partition: {part['name']} (total weight: {part['weight']:.4f})")
        print(f"Number of terms: {len(part['terms'])}")
        
        total_weight = 0
        for k, term in enumerate(part['terms']):
            weight = term['weight']
            total_weight += weight
            print(f"\n  Term {k+1}: weight = {weight:.4f}")
            
            # Print the state vectors
            state_info = term['state_info']
            if part['name'] == 'A|BC':
                print(f"    |ψ_A⟩ = {np.round(state_info['state_A'], 4)}")
                print(f"    |ψ_BC⟩ = {np.round(state_info['state_BC'], 4)}")
            elif part['name'] == 'B|AC':
                print(f"    |ψ_B⟩ = {np.round(state_info['state_B'], 4)}")
                print(f"    |ψ_AC⟩ = {np.round(state_info['state_AC'], 4)}")
            else:  # C|AB
                print(f"    |ψ_C⟩ = {np.round(state_info['state_C'], 4)}")
                print(f"    |ψ_AB⟩ = {np.round(state_info['state_AB'], 4)}")
        
        print(f"\n  Sum of term weights: {total_weight:.6f}")


def reconstruct_state_from_decomposition(decomposition):
    """
    Reconstruct the bi-separable state from its decomposition to verify correctness.
    """
    d = 8
    rho = np.zeros((d, d), dtype=complex)
    
    for part_idx, part in enumerate(decomposition['partitions']):
        for term in part['terms']:
            state_info = term['state_info']
            
            if part['name'] == 'A|BC':
                rho_A = np.outer(state_info['state_A'], state_info['state_A'].conj())
                rho_BC = np.outer(state_info['state_BC'], state_info['state_BC'].conj())
                rho_term = np.kron(rho_A, rho_BC)
                
            elif part['name'] == 'B|AC':
                rho_B = np.outer(state_info['state_B'], state_info['state_B'].conj())
                rho_AC = np.outer(state_info['state_AC'], state_info['state_AC'].conj())
                rho_term = np.kron(rho_B, rho_AC)
                # Permute from B|AC to A|B|C
                perm = np.zeros((8, 8))
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            old_idx = (b << 2) | (a << 1) | c
                            new_idx = (a << 2) | (b << 1) | c
                            perm[new_idx, old_idx] = 1
                rho_term = perm @ rho_term @ perm.T
                
            else:  # C|AB
                rho_C = np.outer(state_info['state_C'], state_info['state_C'].conj())
                rho_AB = np.outer(state_info['state_AB'], state_info['state_AB'].conj())
                rho_term = np.kron(rho_C, rho_AB)
                # Permute from C|AB to A|B|C
                perm = np.zeros((8, 8))
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            old_idx = (c << 2) | (a << 1) | b
                            new_idx = (a << 2) | (b << 1) | c
                            perm[new_idx, old_idx] = 1
                rho_term = perm @ rho_term @ perm.T
            
            rho += decomposition['mixture_weights'][part_idx] * term['weight'] * rho_term
    
    return rho


# Example usage
if __name__ == "__main__":
    # Create a test state (W state with noise)
    w_state = np.zeros(8, dtype=complex)
    w_state[1] = 1/np.sqrt(3)
    w_state[2] = 1/np.sqrt(3)
    w_state[4] = 1/np.sqrt(3)
    rho_W = np.outer(w_state, w_state.conj())
    
    rho_target = 0.8 * rho_W + 0.2 * np.eye(8)/8
    
    # Run optimization
    print("Finding closest bi-separable state...")
    optimizer = ClosestBiSeparableFinder(rho_target, decomposition_length=2, method='SLSQP')
    result = optimizer.optimize(n_restarts=2, max_iter=300)
    
    # Print the decomposition
    print_decomposition(result['decomposition'])
    
    # Verify reconstruction
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    rho_reconstructed = reconstruct_state_from_decomposition(result['decomposition'])
    print(f"Reconstruction error: {np.linalg.norm(result['rho_bi_separable'] - rho_reconstructed):.2e}")
    print(f"Distance to target: {np.sqrt(result['distance']):.6f}")