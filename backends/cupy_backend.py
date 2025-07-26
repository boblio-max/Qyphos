import cupy as cp
from .base_backend import BaseBackend
from ..utils.logger import log

class CuPyBackend(BaseBackend):
    """CuPy-accelerated GPU backend."""

    def __init__(self):
        if not cp.is_available():
            raise RuntimeError("CuPy is not available or no GPU is detected.")
        log.info(f"CuPy Backend Initialized on: {cp.cuda.runtime.getDeviceProperties(0)['name']}")

    @property
    def name(self) -> str:
        return 'cupy'
        
    @property
    def xp(self):
        return cp

    def asarray(self, arr, dtype):
        return cp.asarray(arr, dtype=dtype)
        
    def initial_state(self, n_qubits, dtype):
        state = cp.zeros(2**n_qubits, dtype=getattr(cp, dtype))
        state[0] = 1.0
        return state

    def apply_h(self, state, target_qubit, n_qubits):
        kernel = self._get_h_kernel(n_qubits)
        kernel(state, target_qubit)
        return state

    def apply_x(self, state, target_qubit, n_qubits):
        kernel = self._get_x_kernel(n_qubits)
        kernel(state, target_qubit)
        return state

    def apply_mcx(self, state, controls, target, n_qubits):
        kernel = self._get_mcx_kernel(n_qubits)
        control_mask = 0
        for c in controls:
            control_mask |= (1 << (n_qubits - 1 - c))
        target_bit = 1 << (n_qubits - 1 - target)
        
        kernel(state, control_mask, target_bit)
        return state

    def apply_phase_flip(self, state, indices):
        state[indices] *= -1
        return state

    def apply_diffusion(self, state):
        mean = cp.mean(state)
        state[:] = 2 * mean - state
        return state

    def get_probabilities(self, state):
        return cp.abs(state)**2

    def to_cpu(self, arr):
        return cp.asnumpy(arr)

    # --- CUDA Kernels for performance ---
    @staticmethod
    def _get_h_kernel(n_qubits):
        return cp.RawKernel(r'''
        extern "C" __global__
        void apply_h(complex<double>* state, int k) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            int half_n = 1 << (%(n_qubits)s - k - 1);
            int stride = 1 << (%(n_qubits)s - k);
            
            unsigned int group = i / half_n;
            unsigned int in_group_idx = i %% half_n;
            
            unsigned int idx0 = (group / 2) * stride + in_group_idx;
            unsigned int idx1 = idx0 + half_n;

            if (group %% 2 == 0) {
                 complex<double> val0 = state[idx0];
                 complex<double> val1 = state[idx1];
                 double inv_sqrt2 = 1.0 / sqrt(2.0);
                 
                 state[idx0] = (val0 + val1) * inv_sqrt2;
                 state[idx1] = (val0 - val1) * inv_sqrt2;
            }
        }
        ''' % {'n_qubits': n_qubits-1}, 'apply_h')

    @staticmethod
    def _get_x_kernel(n_qubits):
        return cp.RawKernel(r'''
        extern "C" __global__
        void apply_x(complex<double>* state, int k) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            int bit_k = 1 << (%(n_qubits)s - k - 1);
            if ((i & bit_k) == 0) {
                complex<double> temp = state[i];
                state[i] = state[i + bit_k];
                state[i + bit_k] = temp;
            }
        }
        ''' % {'n_qubits': n_qubits}, 'apply_x')
        
    @staticmethod
    def _get_mcx_kernel(n_qubits):
        return cp.RawKernel(r'''
        extern "C" __global__
        void apply_mcx(complex<double>* state, unsigned int control_mask, unsigned int target_bit) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if ((i & control_mask) == control_mask && (i & target_bit) == 0) {
                complex<double> temp = state[i];
                state[i] = state[i | target_bit];
                state[i | target_bit] = temp;
            }
        }
        ''' % {'n_qubits': n_qubits}, 'apply_mcx')
