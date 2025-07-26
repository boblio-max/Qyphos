import numpy as np
from .circuit import QuantumCircuit

def grover_oracle_circuit(n_qubits: int, solution_indices: list[int]) -> QuantumCircuit:
    """
    Creates the standard Grover oracle which flips the phase of solution states.
    This oracle is decomposed into X, H, and MCX gates.
    """
    circuit = QuantumCircuit(n_qubits)
    all_qubits = list(range(n_qubits))

    for sol_index in solution_indices:
        # Convert integer to binary string and apply X gates to match the state
        binary_str = format(sol_index, f'0{n_qubits}b')
        for i, bit in enumerate(binary_str):
            if bit == '0':
                circuit.x(i)

        # Apply multi-controlled phase flip (H-MCX-H)
        circuit.h(n_qubits - 1)
        if n_qubits > 1:
            circuit.mcx(all_qubits[:-1], all_qubits[-1])
        circuit.h(n_qubits - 1)

        # Uncompute the X gates
        for i, bit in enumerate(binary_str):
            if bit == '0':
                circuit.x(i)
        
        circuit.add_barrier()

    return circuit

def grover_diffusion_circuit(n_qubits: int) -> QuantumCircuit:
    """Creates the standard Grover diffusion operator circuit."""
    circuit = QuantumCircuit(n_qubits)
    all_qubits = list(range(n_qubits))

    for i in all_qubits:
        circuit.h(i)
        circuit.x(i)

    circuit.h(n_qubits - 1)
    if n_qubits > 1:
        circuit.mcx(all_qubits[:-1], all_qubits[-1])
    circuit.h(n_qubits - 1)

    for i in all_qubits:
        circuit.x(i)
        circuit.h(i)

    return circuit
