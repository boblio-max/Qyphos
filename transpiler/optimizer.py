from ..core.circuit import QuantumCircuit, Gate
from typing import List

def optimize_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    A simple transpiler pass to optimize a quantum circuit.
    
    Current Optimizations:
    - Removes adjacent inverse gates (H-H, X-X).
    - Removes barriers.
    """
    optimized_gates: List[Gate] = []
    
    for gate in circuit.gates:
        if not optimized_gates:
            optimized_gates.append(gate)
            continue
        
        last_gate = optimized_gates[-1]
        
        # Rule: H(q) H(q) = I
        if gate[0] == 'h' and last_gate[0] == 'h' and gate[1] == last_gate[1]:
            optimized_gates.pop()
            continue

        # Rule: X(q) X(q) = I
        if gate[0] == 'x' and last_gate[0] == 'x' and gate[1] == last_gate[1]:
            optimized_gates.pop()
            continue
            
        # Rule: Remove barriers
        if gate[0] == 'barrier':
            continue

        optimized_gates.append(gate)
            
    new_circuit = QuantumCircuit(circuit.n_qubits)
    new_circuit.gates = optimized_gates
    new_circuit.metadata = circuit.metadata
    new_circuit.metadata['original_gate_count'] = len(circuit.gates)
    
    return new_circuit
