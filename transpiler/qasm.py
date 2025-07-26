from ..core.circuit import QuantumCircuit

def to_qasm(circuit: QuantumCircuit) -> str:
    """Exports a QuantumCircuit to an OpenQASM 2.0 string."""
    qasm_str = "OPENQASM 2.0;\n"
    qasm_str += 'include "qelib1.inc";\n'
    qasm_str += f"qreg q[{circuit.n_qubits}];\n"
    qasm_str += f"creg c[{circuit.n_qubits}];\n"
    
    for name, qubits, params in circuit.gates:
        if name == 'h':
            qasm_str += f"h q[{qubits[0]}];\n"
        elif name == 'x':
            qasm_str += f"x q[{qubits[0]}];\n"
        elif name == 'mcx':
            controls = ", ".join([f"q[{c}]" for c in qubits[:-1]])
            target = f"q[{qubits[-1]}]"
            qasm_str += f"mcx {controls},{target};\n" # Not standard QASM 2, but common extension
        elif name == 'barrier':
            qasm_str += "barrier q;\n"
        # Skipping functional oracles as they have no direct QASM representation
    
    return qasm_str

def from_qasm(qasm_string: str) -> QuantumCircuit:
    """(Partial Implementation) Imports a QuantumCircuit from an OpenQASM 2.0 string."""
    lines = qasm_string.strip().split('\n')
    n_qubits = 0
    circuit = None

    for line in lines:
        line = line.strip()
        if line.startswith('qreg'):
            n_qubits = int(line.split('[')[1].split(']')[0])
            circuit = QuantumCircuit(n_qubits)
            continue
        if circuit is None: continue

        parts = line.replace(';', '').split()
        if not parts: continue
        
        gate_name = parts[0]
        qubits_str = parts[1]
        
        # Simple parser
        q_indices = [int(q.split('[')[1].split(']')[0]) for q in qubits_str.split(',')]

        if gate_name == 'h':
            circuit.h(q_indices[0])
        elif gate_name == 'x':
            circuit.x(q_indices[0])
        elif gate_name in ['cx', 'CX']:
            circuit.mcx([q_indices[0]], q_indices[1]) # Treat as mcx with 1 control
        elif gate_name == 'ccx':
            circuit.mcx([q_indices[0], q_indices[1]], q_indices[2])
    
    if circuit is None:
        raise ValueError("Invalid QASM string: could not find 'qreg' definition.")
        
    return circuit
