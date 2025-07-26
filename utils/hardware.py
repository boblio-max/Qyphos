import psutil
import cupy
from .logger import log

def get_cpu_memory() -> int:
    """Returns total available system CPU memory in bytes."""
    return psutil.virtual_memory().available

def get_gpu_memory() -> int:
    """Returns total free GPU memory in bytes if CuPy/NVML is available."""
    try:
        cupy.cuda.runtime.getDevice() # Check for device
        free, _ = cupy.cuda.runtime.memGetInfo()
        return free
    except Exception:
        return 0

def check_memory_requirements(n_qubits: int, precision: str = 'complex128', backend: str = 'cpu'):
    """
    Checks if the system has enough memory for the statevector.

    Args:
        n_qubits: The number of qubits for the simulation.
        precision: The data type ('complex128' or 'complex64').
        backend: The target backend ('cpu' or 'gpu').

    Raises:
        MemoryError: If the estimated memory exceeds 80% of available memory.
    """
    bytes_per_element = 16 if precision == 'complex128' else 8
    required_memory = (2 ** n_qubits) * bytes_per_element
    
    if backend == 'gpu':
        available_memory = get_gpu_memory()
        if available_memory == 0:
            raise EnvironmentError("GPU backend selected, but no compatible GPU found or CuPy failed.")
        device = "GPU VRAM"
    else:
        available_memory = get_cpu_memory()
        device = "CPU RAM"

    # Use 85% of available memory as a safe threshold
    safe_threshold = 0.85 * available_memory
    
    log.info(
        f"Memory Check: Required for statevector: {required_memory / 1e9:.2f} GB. "
        f"Available {device}: {available_memory / 1e9:.2f} GB."
    )
    
    if required_memory > safe_threshold:
        raise MemoryError(
            f"Insufficient memory for {n_qubits}-qubit simulation. "
            f"Required: {required_memory/1e9:.2f} GB, "
            f"Available ({device}): {available_memory/1e9:.2f} GB. "
            f"Try reducing the number of qubits."
        )
