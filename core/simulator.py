import time
from typing import List, Optional, Callable
import numpy as np

from .circuit import QuantumCircuit
from ..backends import get_backend
from ..utils.hardware import check_memory_requirements
from ..utils.logger import log
from ..noise.models import apply_stochastic_kraus, KrausOperators

class QyphosSimulator:
    """
    A high-performance, matrix-free quantum circuit simulator.
    """
    def __init__(self, backend_name: str = 'auto', precision: str = 'complex128'):
        self.backend = get_backend(backend_name)
        self.precision = precision
        self.statevector = None
        self.n_qubits = 0
        self.noise_model: Optional[KrausOperators] = None

    def _initialize_statevector(self, n_qubits: int):
        """Prepares the statevector for simulation."""
        check_memory_requirements(n_qubits, self.precision, self.backend.name)
        self.n_qubits = n_qubits
        self.statevector = self.backend.initial_state(n_qubits, self.precision)
        log.info(f"Initialized {n_qubits}-qubit statevector on '{self.backend.name}' backend.")

    def add_noise(self, noise_model: KrausOperators):
        """Adds a noise model to be applied after each gate."""
        self.noise_model = noise_model
        log.info(f"Noise model added to the simulator.")

    def run(self, circuit: QuantumCircuit, store_history: bool = False) -> dict:
        """
        Executes a quantum circuit.

        Args:
            circuit: The QuantumCircuit to simulate.
            store_history: If True, stores the statevector at each step.

        Returns:
            A dictionary containing results and performance metrics.
        """
        start_time = time.perf_counter()
        if self.statevector is None or self.n_qubits != circuit.n_qubits:
            self._initialize_statevector(circuit.n_qubits)
        else: # Reset state if running again
            self.statevector.fill(0)
            self.statevector[0] = 1.0

        state_history = []
        if store_history:
            state_history.append(self.statevector.copy())

        for i, (name, qubits, params) in enumerate(circuit.gates):
            # --- Apply Gate Logic ---
            if name == 'h':
                self.backend.apply_h(self.statevector, qubits[0], self.n_qubits)
            elif name == 'x':
                self.backend.apply_x(self.statevector, qubits[0], self.n_qubits)
            elif name == 'mcx':
                controls, target = qubits[:-1], qubits[-1]
                self.backend.apply_mcx(self.statevector, controls, target, self.n_qubits)
            elif name == 'index_oracle':
                indices = self.backend.asarray(params['indices'], dtype=np.int64)
                self.backend.apply_phase_flip(self.statevector, indices)
            elif name == 'diffusion':
                self.backend.apply_diffusion(self.statevector)
            elif name == 'functional_oracle':
                self._apply_functional_oracle(params['func'])
            
            # --- Apply Noise ---
            if self.noise_model and name not in ['barrier', 'index_oracle', 'diffusion']:
                # Note: Applying noise stochastically to entire statevector is an approximation.
                # A more accurate model would be a density matrix simulation.
                self.statevector = apply_stochastic_kraus(self.statevector, self.noise_model, self.backend)

            if store_history and name != 'barrier':
                state_history.append(self.statevector.copy())

        end_time = time.perf_counter()
        
        final_probs = self.get_probabilities()
        
        results = {
            "final_statevector": self.statevector,
            "final_probabilities": self.backend.to_cpu(final_probs),
            "state_history": [self.backend.to_cpu(s) for s in state_history],
            "simulation_time_sec": end_time - start_time,
            "n_qubits": self.n_qubits,
            "gate_count": len(circuit.gates),
            "backend": self.backend.name
        }
        return results
        
    def _apply_functional_oracle(self, oracle_func: Callable[[int], bool]):
        """A simple, but slow, way to implement a functional oracle."""
        if self.backend.name == 'cupy':
            raise NotImplementedError("Functional oracles are not yet optimized for GPU backend.")
        
        marked_indices = [i for i in range(2**self.n_qubits) if oracle_func(i)]
        self.backend.apply_phase_flip(self.statevector, marked_indices)

    def get_probabilities(self) -> np.ndarray:
        """Returns the probability distribution of the current state."""
        if self.statevector is None:
            raise RuntimeError("Run a circuit before getting probabilities.")
        return self.backend.get_probabilities(self.statevector)

    def measure(self, n_shots: int = 1024) -> List[int]:
        """Simulates measurement of the final state."""
        probs = self.backend.to_cpu(self.get_probabilities())
        # Normalize to counteract potential floating-point errors
        probs /= np.sum(probs)
        return list(np.random.choice(2**self.n_qubits, size=n_shots, p=probs))
