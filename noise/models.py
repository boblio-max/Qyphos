import numpy as np
from typing import List, Tuple

# Kraus Operators are represented as a list of numpy arrays
KrausOperators = List[np.ndarray]

def apply_stochastic_kraus(state: np.ndarray, kraus_ops: KrausOperators, backend) -> np.ndarray:
    """
    Applies a Kraus map to a statevector stochastically.
    
    Args:
        state: The current statevector.
        kraus_ops: A list of Kraus operators for the channel.
        backend: The simulation backend.

    Returns:
        The new statevector after a randomly chosen Kraus operator is applied.
    """
    xp = backend.xp
    
    # Calculate probabilities p_k = <psi| E_k^dagger E_k |psi>
    probabilities = []
    for Ek in kraus_ops:
        Ek_gpu = backend.asarray(Ek, dtype=state.dtype)
        state_after_Ek = Ek_gpu @ state
        prob = xp.vdot(state_after_Ek, state_after_Ek).real
        probabilities.append(prob)

    # Choose an operator based on probabilities
    chosen_index = np.random.choice(len(kraus_ops), p=np.asarray(probabilities))
    
    # Apply the chosen operator and re-normalize
    chosen_Ek = backend.asarray(kraus_ops[chosen_index], dtype=state.dtype)
    new_state = chosen_Ek @ state
    norm = xp.linalg.norm(new_state)
    return new_state / norm


def depolarizing_channel(p: float) -> KrausOperators:
    """
    Creates Kraus operators for a single-qubit depolarizing channel.
    
    Args:
        p: The probability of depolarization.
    
    Returns:
        A list of Kraus operators.
    """
    I = np.identity(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p / 3) * X
    E2 = np.sqrt(p / 3) * Y
    E3 = np.sqrt(p / 3) * Z
    
    return [E0, E1, E2, E3]

def amplitude_damping_channel(gamma: float) -> KrausOperators:
    """
    Creates Kraus operators for a single-qubit amplitude damping channel.
    
    Args:
        gamma: The damping probability (related to T1 time).
    
    Returns:
        A list of Kraus operators.
    """
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    
    return [E0, E1]
