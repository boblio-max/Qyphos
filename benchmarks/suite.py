import time
import numpy as np
import pandas as pd
from ..core.simulator import QyphosSimulator
from ..cli import build_grover_circuit # Re-use the circuit builder
from ..utils.logger import log

# This is the original code, saved for comparison
class LegacyGrover:
    def __init__(self, n_qubits, solution_indices):
        self.n_qubits = n_qubits
        self.n_items = 2**n_qubits
        self.solution_indices = solution_indices
        self.oracle_operator = self._build_oracle_operator()
        self.diffusion_operator = self._build_diffusion_operator()
        self.grover_operator = self.diffusion_operator @ self.oracle_operator
    
    def _build_oracle_operator(self):
        op = np.identity(self.n_items, dtype=complex)
        op[self.solution_indices, self.solution_indices] = -1
        return op
        
    def _build_diffusion_operator(self):
        initial_state = np.ones(self.n_items, dtype=complex) / np.sqrt(self.n_items)
        return 2 * np.outer(initial_state, initial_state.conj()) - np.identity(self.n_items, dtype=complex)

    def run(self, iterations):
        state = np.ones(self.n_items, dtype=complex) / np.sqrt(self.n_items)
        for _ in range(iterations):
            state = self.grover_operator @ state
        return state

def run_benchmark_suite():
    """Compares the new Qyphos simulator with the legacy matrix-based version."""
    
    benchmark_cases = [
        # (n_qubits, n_solutions)
        (4, 1),
        (6, 1),
        (8, 2),
        (10, 4),
        (12, 1), # Max for legacy
        (16, 1), # Qyphos only
        (20, 4), # Qyphos only
        (24, 10),# Qyphos only
    ]
    
    results = []
    
    for n_qubits, n_sols in benchmark_cases:
        log.info(f"\n--- BENCHMARKING: {n_qubits} Qubits, {n_sols} Solutions ---")
        solutions = list(range(n_sols))
        n_items = 2**n_qubits
        optimal_iters = int(np.round(np.pi / 4 * np.sqrt(n_items / n_sols)))

        # --- Legacy Matrix-Based Run ---
        if n_qubits <= 12:
            try:
                start_time = time.perf_counter()
                legacy_sim = LegacyGrover(n_qubits, solutions)
                legacy_sim.run(optimal_iters)
                end_time = time.perf_counter()
                legacy_time = end_time - start_time
                log.info(f"[Legacy] Time: {legacy_time:.6f} s")
            except MemoryError:
                legacy_time = np.inf
                log.warning("[Legacy] MemoryError: Could not run.")
        else:
            legacy_time = np.nan
            log.info("[Legacy] Skipped (too many qubits).")
            
        # --- New Qyphos Run (CPU) ---
        try:
            sim_cpu = QyphosSimulator('numpy')
            circuit, _ = build_grover_circuit(n_qubits, solutions)
            start_time = time.perf_counter()
            sim_cpu.run(circuit)
            end_time = time.perf_counter()
            qyphos_cpu_time = end_time - start_time
            log.info(f"[Qyphos CPU] Time: {qyphos_cpu_time:.6f} s")
        except MemoryError:
            qyphos_cpu_time = np.inf
            log.warning("[Qyphos CPU] MemoryError: Could not run.")

        # --- New Qyphos Run (GPU) ---
        try:
            sim_gpu = QyphosSimulator('cupy')
            circuit, _ = build_grover_circuit(n_qubits, solutions)
            start_time = time.perf_counter()
            sim_gpu.run(circuit)
            end_time = time.perf_counter()
            qyphos_gpu_time = end_time - start_time
            log.info(f"[Qyphos GPU] Time: {qyphos_gpu_time:.6f} s")
        except (RuntimeError, MemoryError):
            qyphos_gpu_time = np.nan
            log.info("[Qyphos GPU] Skipped (CuPy not available or MemoryError).")
            
        results.append({
            "Qubits": n_qubits,
            "Solutions": n_sols,
            "Iterations": optimal_iters,
            "Legacy Time (s)": legacy_time,
            "Qyphos CPU Time (s)": qyphos_cpu_time,
            "Qyphos GPU Time (s)": qyphos_gpu_time
        })
        
    df = pd.DataFrame(results)
    df['Speedup (CPU vs Legacy)'] = df['Legacy Time (s)'] / df['Qyphos CPU Time (s)']
    df['Speedup (GPU vs CPU)'] = df['Qyphos CPU Time (s)'] / df['Qyphos GPU Time (s)']
    
    print("\n\n--- BENCHMARK RESULTS ---")
    print(df.to_string(index=False, float_format="%.4f"))

if __name__ == '__main__':
    run_benchmark_suite()
