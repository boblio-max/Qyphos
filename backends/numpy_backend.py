import numpy as np
from numba import njit, prange
from .base_backend import BaseBackend

@njit(cache=True)
def _apply_h_numba(state, k, n_qubits):
    """Numba-accelerated Hadamard gate."""
    n_states = 1 << n_qubits
    half_n = 1 << (n_qubits - k - 1)
    stride = 1 << (n_qubits - k)
    
    inv_sqrt2 = 1 / np.sqrt(2)
    
    for i in prange(n_states // stride):
        for j in prange(half_n):
            idx0 = i * stride + j
            idx1 = idx0 + half_n
            
            val0 = state[idx0]
            val1 = state[idx1]
            
            state[idx0] = (val0 + val1) * inv_sqrt2
            state[idx1] = (val0 - val1) * inv_sqrt2
    return state
    
@njit(cache=True)
def _apply_x_numba(state, k, n_qubits):
    """Numba-accelerated X gate."""
    n_states = 1 << n_qubits
    bit_k = 1 << (n_qubits - k - 1)
    for i in prange(n_states):
        if (i & bit_k) == 0:
            state[i], state[i + bit_k] = state[i + bit_k], state[i]
    return state

@njit(cache=True)
def _apply_mcx_numba(state, controls, target, n_qubits):
    """Numba-accelerated multi-controlled X gate."""
    control_mask = 0
    for c in controls:
        control_mask |= (1 << (n_qubits - 1 - c))
        
    target_bit = 1 << (n_qubits - 1 - target)
    
    for i in prange(len(state)):
        if (i & control_mask) == control_mask:
            if (i & target_bit) == 0: # Apply X to |...0...> part of pair
                 state[i], state[i | target_bit] = state[i | target_bit], state[i]
                 
@njit(cache=True)
def _apply_diffusion_numba(state):
    """Numba-accelerated diffusion operator."""
    mean = state.mean()
    state[:] = 2 * mean - state
    # Restore norm (optional, but good practice)
    # norm = np.sqrt(np.sum(np.abs(state)**2))
    # state /= norm
    return state

class NumPyBackend(BaseBackend):
    """Numba-accelerated CPU backend."""
    
    @property
    def name(self) -> str:
        return 'numpy'
        
    @property
    def xp(self):
        return np

    def asarray(self, arr, dtype):
        return np.asarray(arr, dtype=dtype)
        
    def initial_state(self, n_qubits, dtype):
        state = np.zeros(2**n_qubits, dtype=getattr(np, dtype))
        state[0] = 1.0
        return state

    def apply_h(self, state, target_qubit, n_qubits):
        return _apply_h_numba(state, target_qubit, n_qubits)

    def apply_x(self, state, target_qubit, n_qubits):
        return _apply_x_numba(state, target_qubit, n_qubits)

    def apply_mcx(self, state, controls, target, n_qubits):
        return _apply_mcx_numba(state, controls, target, n_qubits)
        
    def apply_phase_flip(self, state, indices):
        state[indices] *= -1
        return state

    def apply_diffusion(self, state):
        return _apply_diffusion_numba(state)

    def get_probabilities(self, state):
        return np.abs(state)**2

    def to_cpu(self, arr):
        return np.asarray(arr)
