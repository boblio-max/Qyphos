Qyphos Quantum Search Simulator
################################################################################
#
#    ██████╗ ██╗   ██╗██████╗ ██╗  ██╗ ██████╗ ███████╗
#   ██╔═══██╗╚██╗ ██╔╝██╔══██╗██║  ██║██╔═══██╗██╔════╝
#   ██║   ██║ ╚████╔╝ ██████╔╝███████║██║   ██║███████╗
#   ██║  ███║  ╚██╔╝  ██╔═══╝ ██╔══██║██║   ██║╚════██║
#   ╚█████ ██║  ██║   ██║     ██║  ██║╚██████╔╝███████║
#    ╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
#
################################################################################
Qyphos Core: A Next-Generation Quantum Search & Estimation Framework

Qyphos is a high-performance, research-grade quantum algorithm simulator built from the ground up in Python. Originally a simple matrix-based demonstration of Grover's algorithm, it has been completely re-architected to be a powerful, scalable, and extensible framework for simulating quantum search and related algorithms on classical hardware.

It is designed for students, researchers, and engineers who need to simulate quantum circuits beyond the 15-qubit limit of matrix-based approaches, with support for realistic noise, hardware acceleration, and modern quantum development workflows.

🚀 Core Features
Qyphos is more than just a simulator; it's a complete toolkit for quantum algorithm research.

🧠 High-Performance Matrix-Free Engine: Simulates circuits by applying gate logic directly to the statevector, reducing memory complexity from O(4 
n
 ) to O(2 
n
 ). Easily simulates 30+ qubits on a standard machine.

⚡ Pluggable CPU/GPU Backends: Automatically uses CuPy for NVIDIA GPU acceleration if available, falling back to a Numba JIT-compiled CPU backend for maximum performance on any hardware.

📊 Interactive Dashboard: A built-in Streamlit GUI for real-time, interactive simulation and visualization of state evolution, interference patterns, and success probabilities.

🔌 Realistic Noise Models: Go beyond ideal simulations. Inject stochastic noise channels like depolarizing and amplitude damping after each gate to model decoherence on real NISQ devices.

✈️ Quantum Transpiler & Optimizer: Features a built-in compiler pass that optimizes quantum circuits by merging adjacent inverse gates (H-H → I) and removing redundant operations, reducing circuit depth and improving fidelity.

🤝 OpenQASM 2.0 Interoperability: Export and import circuits to/from the OpenQASM standard, enabling seamless integration with frameworks like Qiskit, Cirq, and real quantum hardware backends.

💻 Powerful CLI & Python API: Use Qyphos as a command-line tool for quick experiments or import it as a library (import qyphos) into your own research projects for programmatic control.

🔮 Pluggable & Synthesizable Oracles: Define oracles by simply providing a list of solution indices or write a Python function (lambda x: ...) that Qyphos automatically synthesizes into a quantum oracle.

⏱️ Built-in Benchmark Suite: A dedicated module to benchmark the matrix-free engine against the legacy matrix-based approach, providing concrete stats on speedup and resource usage.

🛠️ Installation
Get up and running with Qyphos in a few simple steps. A virtual environment is highly recommended.

1. Clone the repository:

Bash

git clone https://github.com/your-username/qyphos.git
cd qyphos
2. Create and activate a virtual environment:

Bash

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
3. Install dependencies:
All standard dependencies are listed in requirements.txt.

Bash

pip install -r requirements.txt
4. (Optional) GPU Acceleration:
For massive performance gains on NVIDIA GPUs, install CuPy. Your CuPy version must match your system's CUDA Toolkit version.

Find your CUDA version: nvcc --version

Install the corresponding CuPy wheel. For example, for CUDA 12.x:

Bash

pip install cupy-cuda12x
See the CuPy installation guide for more details.

💡 Usage
Qyphos can be used in three main ways: via the Interactive Dashboard, the Command-Line Interface, or as a Python library.

1. Interactive Dashboard
The easiest way to get started. Launch the Streamlit application for a full GUI experience.

Bash

streamlit run qyphos/app.py
This will open a new tab in your browser where you can configure and run simulations interactively.

2. Command-Line Interface (CLI)
For scripting and quick runs, use the built-in CLI.

Bash

# Run a 10-qubit search for solutions 42 and 512 on the GPU
python -m qyphos.cli run --qubits 10 --solutions 42 512 --backend cupy

# Run a 20-qubit search without transpiler optimizations
python -m qyphos.cli run --qubits 20 --solutions 1048570 --no-optimize
3. Python API
For maximum flexibility, import Qyphos directly into your Python projects.

Python

from qyphos.core.simulator import QyphosSimulator
from qyphos.cli import build_grover_circuit # Helper for Grover
from qyphos.noise.models import depolarizing_channel

# 1. Define the search problem
n_qubits = 16
solutions = [12345, 54321]

# 2. Build the quantum circuit for Grover's algorithm
# This helper function calculates optimal iterations and assembles the circuit.
grover_circuit, optimal_iters = build_grover_circuit(n_qubits, solutions)
print(f"Built a {n_qubits}-qubit circuit with {len(grover_circuit)} gates.")

# 3. Initialize the simulator (auto-detects GPU)
simulator = QyphosSimulator(backend_name='auto')

# 4. (Optional) Add a noise model
noise = depolarizing_channel(p=0.001) # 0.1% depolarizing error
simulator.add_noise(noise)

# 5. Run the simulation
results = simulator.run(grover_circuit)

# 6. Analyze the output
print(f"Simulation finished in {results['simulation_time_sec']:.4f}s.")
final_probs = results['final_probabilities']
success_prob = sum(final_probs[s] for s in solutions)
print(f"Total probability of finding a solution: {success_prob:.4%}")
벤치마크 (Benchmarks)
To see the dramatic performance improvement of the new matrix-free engine, run the built-in benchmark suite. This will compare the legacy matrix-based Grover against the new Numba (CPU) and CuPy (GPU) backends.

Bash

python -m qyphos.benchmarks.suite
Example Output:

--- BENCHMARK RESULTS ---
 Qubits  Solutions  Iterations  Legacy Time (s)  Qyphos CPU Time (s)  Qyphos GPU Time (s)  Speedup (CPU vs Legacy)  Speedup (GPU vs CPU)
      4          1           1           0.0004               0.0041                  nan                   0.0913                   nan
      6          1           6           0.0018               0.0044                  nan                   0.4079                   nan
      8          2           9           0.0195               0.0053                  nan                   3.6653                   nan
     10          4          12           0.3698               0.0076                  nan                  48.9134                   nan
     12          1          45          22.0673               0.0152                  nan                1455.5902                   nan
     16          1         181              nan               0.1171               0.0249                      nan                4.7081
     20          4         355              nan               1.8893               0.2015                      nan                9.3745
     24         10        1137              nan              36.9501               2.8872                      nan               12.7975
Note: The legacy simulator becomes unusable due to MemoryError around 12-14 qubits. The Qyphos engine scales linearly in time and memory, enabling much larger simulations.

🏗️ Project Structure
The codebase is organized into modular components for clarity and extensibility.

qyphos/
├── app.py                  # Streamlit dashboard application
├── backends/               # CPU/GPU simulation backends
├── benchmarks/             # Performance comparison suite
├── cli.py                  # Command-line interface
├── core/                   # Core components: Simulator, Circuit, Oracles
├── noise/                  # Realistic noise models (Kraus operators)
├── transpiler/             # Quantum circuit optimizer and QASM parser
└── utils/                  # Hardware detection and logging utilities
🤝 Contributing
Contributions are welcome! Whether it's reporting a bug, proposing a new feature, or submitting a pull request, your input is valued. Please open an issue on the GitHub repository to get started.
