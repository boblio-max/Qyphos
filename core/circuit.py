from typing import List, Tuple, Union, Optional, Callable

# Gate: (name, target_qubits, params)
Gate = Tuple[str, Union[int, List[int]], Optional[dict]]

class QuantumCircuit:
    """
    Represents a quantum circuit as a sequence of gates.
    """
    def __init__(self, n_qubits: int):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")
        self.n_qubits = n_qubits
        self.gates: List[Gate] = []
        self.metadata = {}

    def h(self, qubit: int):
        self.gates.append(('h', [qubit], {}))

    def x(self, qubit: int):
        self.gates.append(('x', [qubit], {}))

    def mcx(self, controls: List[int], target: int):
        self.gates.append(('mcx', controls + [target], {}))

    def oracle(self, oracle_func: Callable[[int], bool]):
        """A functional oracle definition."""
        self.gates.append(('functional_oracle', [], {'func': oracle_func}))

    def index_oracle(self, solution_indices: List[int]):
        """An oracle that marks specific indices."""
        self.gates.append(('index_oracle', [], {'indices': solution_indices}))

    def diffusion(self):
        self.gates.append(('diffusion', list(range(self.n_qubits)), {}))

    def __len__(self) -> int:
        return len(self.gates)

    def __str__(self) -> str:
        header = f"QuantumCircuit({self.n_qubits} qubits, {len(self.gates)} gates)"
        gate_str = "\n".join([f"  {g[0]:<10} {g[1]}" for g in self.gates])
        return f"{header}\n{gate_str}"

    def add_barrier(self):
        self.gates.append(('barrier', [], {}))
