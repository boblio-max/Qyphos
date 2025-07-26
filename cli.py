import argparse
import numpy as np
from .core.simulator import QyphosSimulator
from .core.circuit import QuantumCircuit
from .transpiler.optimizer import optimize_circuit
from .utils.logger import log

def build_grover_circuit(n_qubits, solutions):
    """Helper to build a full Grover circuit."""
    # Optimal iterations
    n_items = 2**n_qubits
    n_sols = len(solutions)
    if n_sols == 0: return QuantumCircuit(n_qubits), 0
    
    optimal_iters = int(np.round(np.pi / 4 * np.sqrt(n_items / n_sols)))
    
    circuit = QuantumCircuit(n_qubits)
    # 1. Superposition
    for i in range(n_qubits):
        circuit.h(i)
    
    # 2. Grover iterations
    for _ in range(optimal_iters):
        circuit.add_barrier()
        circuit.index_oracle(solutions)
        circuit.diffusion()
    
    return circuit, optimal_iters

def main():
    parser = argparse.ArgumentParser(description="Qyphos: A High-Performance Quantum Simulator.")
    parser.add_argument('command', choices=['run', 'benchmark'], help="Command to execute.")
    parser.add_argument('--qubits', type=int, required=True, help="Number of qubits in the simulation.")
    parser.add_argument('--solutions', type=int, nargs='+', required=True, help="List of solution indices.")
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'numpy', 'cupy'], help="Simulation backend.")
    parser.add_argument('--no-optimize', action='store_true', help="Disable the transpiler optimization pass.")
    
    args = parser.parse_args()

    if args.command == 'run':
        log.info(f"Starting Grover's search for {args.solutions} in a {args.qubits}-qubit space.")
        
        # Build circuit
        circuit, optimal_iters = build_grover_circuit(args.qubits, args.solutions)
        log.info(f"Constructed Grover circuit with {len(circuit)} gates for {optimal_iters} optimal iterations.")

        # Optimize circuit
        if not args.no_optimize:
            original_gates = len(circuit.gates)
            circuit = optimize_circuit(circuit)
            log.info(f"Transpiler optimized circuit: {original_gates} -> {len(circuit.gates)} gates.")

        # Run simulation
        simulator = QyphosSimulator(backend_name=args.backend)
        results = simulator.run(circuit)
        
        # Analyze and print results
        log.info(f"Simulation finished in {results['simulation_time_sec']:.4f} seconds on '{results['backend']}' backend.")
        
        probs = results['final_probabilities']
        success_prob = np.sum(probs[args.solutions])
        
        log.info(f"Total probability of finding a solution: {success_prob:.4f}")
        
        most_likely_index = np.argmax(probs)
        log.info(f"Most likely outcome: index {most_likely_index} with probability {probs[most_likely_index]:.4f}")
        if most_likely_index in args.solutions:
            log.info("✅ Most likely outcome is a correct solution.")
        else:
            log.info("❌ Most likely outcome is NOT a correct solution.")

if __name__ == '__main__':
    main()
