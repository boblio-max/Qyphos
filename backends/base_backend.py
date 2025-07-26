from abc import ABC, abstractmethod
from typing import Any, Tuple

class BaseBackend(ABC):
    """Abstract base class for quantum simulation backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend (e.g., 'numpy', 'cupy')."""
        pass

    @property
    @abstractmethod
    def xp(self) -> Any:
        """The numpy-like module (numpy, cupy, etc.)."""
        pass
        
    @abstractmethod
    def asarray(self, arr: Any, dtype: Any) -> Any:
        """Converts an array-like object to the backend's array type."""
        pass
        
    @abstractmethod
    def initial_state(self, n_qubits: int, dtype: str) -> Any:
        """Creates the initial |0...0> state."""
        pass
        
    @abstractmethod
    def apply_h(self, state: Any, target_qubit: int, n_qubits: int):
        """Applies a Hadamard gate."""
        pass

    @abstractmethod
    def apply_x(self, state: Any, target_qubit: int, n_qubits: int):
        """Applies a Pauli-X gate."""
        pass

    @abstractmethod
    def apply_mcx(self, state: Any, controls: list[int], target: int, n_qubits: int):
        """Applies a multi-controlled X gate."""
        pass
        
    @abstractmethod
    def apply_phase_flip(self, state: Any, indices: Any):
        """Flips the phase of specified computational basis states."""
        pass

    @abstractmethod
    def apply_diffusion(self, state: Any):
        """Applies the Grover diffusion operator (inversion about the mean)."""
        pass

    @abstractmethod
    def get_probabilities(self, state: Any) -> Any:
        """Calculates measurement probabilities from a statevector."""
        pass

    @abstractmethod
    def to_cpu(self, arr: Any) -> Any:
        """Moves an array from the device to host CPU memory."""
        pass
