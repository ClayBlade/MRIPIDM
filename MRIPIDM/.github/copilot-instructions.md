# Copilot Instructions for MRIPIDM

## Project Overview
MRIPIDM is a GPU-accelerated MRI simulation toolkit, primarily using CUDA C++ for high-performance spin evolution and signal simulation. The codebase integrates with Python for data preparation and uses the nlohmann::json library for configuration and data exchange.

## Key Components
- **CUDA Kernels**: Main simulation logic is in `BlochKernelGMGPU.cu` and called from `DoScanAtGPU.cu`.
- **Host Control**: `DoScanAtGPU.cu` orchestrates memory allocation, data transfer, and kernel launches.
- **Python Data Prep**: `utils/labeledSpace.py` generates `.pkl` files with simulation parameters and magnetization vectors, which are later converted to JSON.
- **JSON Handling**: `json.hpp` (nlohmann::json) is used for reading simulation parameters and initial conditions.

## Developer Workflows
- **Build**: No build scripts are present; compile CUDA code with `nvcc` (e.g., `nvcc -Xcompiler -fopenmp -o DoScanAtGPU.exe DoScanAtGPU.cu BlochKernelGMGPU.cu`).
- **Python Preprocessing**: Run `utils/convertpkltojson.py` to convert `.pkl` files to JSON for C++ consumption.
- **Simulation Run**: The main entry is `DoScanAtGPU.cu`, which expects a JSON file as input (see hardcoded path in source).
- **Debugging**: Use CUDA-aware debuggers (e.g., Nsight, cuda-gdb) for GPU code. Host-side logic can be debugged with standard C++ tools.

## Project-Specific Conventions
- **Memory Management**: Host/device memory is managed manually; always ensure proper allocation and `cudaMemcpy` usage.
- **Data Layout**: Multi-dimensional arrays are flattened; index calculations are explicit and must match between host and device.
- **Parameter Passing**: Most simulation parameters are passed as pointers, not structs.
- **Default Values**: Many simulation parameters are initialized with hardcoded defaults in `DoScanAtGPU.cu`.
- **Error Handling**: CUDA errors are checked with `cudaSuccess` and early returns; no exception handling is used.

## Integration Points
- **Python ↔ C++**: Data flows from Python (`labeledSpace.py`) → `.pkl` → JSON → C++ (`DoScanAtGPU.cu`).
- **External Libraries**: Uses nlohmann::json (header-only, `json.hpp`). Optionally supports Intel IPP or AMD Framewave for signal processing (see preprocessor flags `IPP`/`FW`).
- **GPU Selection**: GPU device is selected via `cudaSetDevice` using a pointer parameter.

## Examples
- **Launching a Simulation**: See the `main()` in `DoScanAtGPU.cu` for the full workflow: load JSON, allocate memory, transfer data, launch kernel, fetch results.
- **Data Preparation**: See `utils/labeledSpace.py` for how simulation input data is structured and saved.

## Tips for AI Agents
- Always check for hardcoded paths and update as needed for your environment.
- When adding new simulation parameters, update both Python and C++ sides for consistency.
- Follow the explicit memory allocation and pointer-passing style used throughout the codebase.
- For new kernels, ensure grid/block configuration logic matches the data layout and device capabilities.

---
_Last updated: July 2025_
