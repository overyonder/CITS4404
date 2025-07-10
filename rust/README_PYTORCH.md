# PyTorch Engine Integration

This document explains how to use the PyTorch engine for neural network evolution in the CITS4404 project.

## Overview

The PyTorch engine provides CUDA-accelerated neural network computation using the mature PyTorch ecosystem. This engine offers:

- **CUDA Acceleration**: Leverages PyTorch's optimized CUDA kernels
- **Mature Ecosystem**: Benefits from PyTorch's extensive optimizations
- **Research-Ready**: Easily extensible for advanced ML research
- **Automatic Memory Management**: PyTorch handles GPU memory automatically

## Installation Requirements

### 1. Install PyTorch C++ Library (libtorch)

#### Windows:
1. Download libtorch from https://pytorch.org/get-started/locally/
2. Choose: `PyTorch Build=Stable`, `OS=Windows`, `Package=LibTorch`, `Compute=CUDA 11.8` (or CPU)
3. Extract to `C:\libtorch` (or preferred location)
4. Set environment variable: `LIBTORCH=C:\libtorch`

#### Linux/macOS:
```bash
# Download and extract libtorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-*.zip
export LIBTORCH=/path/to/libtorch
```

### 2. Build with PyTorch Support

```bash
# Enable torch feature during compilation
cargo build --features torch

# Run with PyTorch engine
cargo run --features torch -- --engine torch

# For TUI mode with PyTorch support
cargo run --features torch
```

## Usage Examples

### Command Line Interface

```bash
# Train using PyTorch engine
cargo run --features torch -- \
    --engine torch \
    --generations 100 \
    --population-size 256 \
    --save-to pytorch_model.json

# Available engine options: cpu, gpu, torch
cargo run --features torch -- --engine torch --help
```

### TUI Interface

1. Start the TUI with PyTorch support:
   ```bash
   cargo run --features torch
   ```

2. Navigate to Configuration tab
3. Use arrow keys to select "Engine" parameter
4. Press Left/Right arrows to cycle: `CPU → GPU → PyTorch → CPU`
5. Configure other parameters as needed
6. Start training

### Engine Selection in TUI

The engine selection cycles through available options:
- **CPU**: Fast, deterministic, good for debugging
- **GPU**: WebGPU-based, cross-platform GPU acceleration  
- **PyTorch**: CUDA-accelerated, research-ready (only if `torch` feature enabled)

## Performance Comparison

| Engine | Setup | Dependencies | Performance | Platform |
|--------|-------|--------------|-------------|----------|
| CPU | None | None | Good | Universal |
| GPU | None | GPU drivers | Very Good | GPU required |
| PyTorch | libtorch | ~2GB install | Excellent | CUDA preferred |

## Technical Details

### Architecture

The `TorchIndividual` implements the same `Individual` trait as other engines:

```rust
use crate::engines::TorchIndividual;

// Same interface as StackIndividual and GpuIndividual
let individual = TorchIndividual::default();
let output = individual.forward_propagate(&input, activation);
```

### Threading Safety

PyTorch modules are wrapped in `Arc<Mutex<>>` for thread safety:
```rust
pub struct TorchIndividual {
    device: Device,
    model: Arc<Mutex<nn::Sequential>>,  // Thread-safe PyTorch model
    vs: Arc<Mutex<nn::VarStore>>,       // Thread-safe parameters
    weights_cache: Vec<f32>,            // CPU cache for genetic ops
}
```

### Memory Management

- **Hybrid Strategy**: CPU cache for genetics, GPU for computation
- **Automatic Sync**: Weights synchronized between CPU/GPU as needed
- **Error Handling**: Graceful fallbacks if GPU operations fail

## Troubleshooting

### Quick Diagnostic

First, run the diagnostic script:
```cmd
check_torch_install.cmd
```

This will verify your installation and point out any issues.

### Common Issues

1. **"Cannot find libtorch install"**
   ```bash
   # Set environment variable
   export LIBTORCH=/path/to/libtorch
   # or bypass version check
   export LIBTORCH_BYPASS_VERSION_CHECK=1
   ```

2. **"Cannot open include file: 'torch/torch.h'"**
   - **Cause**: Downloaded Python PyTorch instead of C++ libtorch
   - **Solution**: Download the **C++** version from https://pytorch.org/
   - **Check**: Verify `C:\libtorch\include\torch\torch.h` exists

3. **Linker errors: "cannot open input file 'torch_cpu.lib'"**
   - **Cause**: Incorrect library paths in build script
   - **Solution**: Use `build_torch_clean.cmd` for clean rebuild
   - **Manual fix**: Set `LIBTORCH_LIB=C:\libtorch\lib`

4. **Version mismatch warnings**
   - **Cause**: PyTorch 2.7.1 vs expected 2.7.0
   - **Solution**: Set `LIBTORCH_BYPASS_VERSION_CHECK=1`
   - **Safe**: Minor version differences are usually compatible

5. **Double path issues (C:\libtorch\lib\lib)**
   - **Cause**: Build script bug in torch-sys crate
   - **Solution**: Use the provided batch scripts with correct env vars

6. **CUDA out of memory**
   - Reduce population size
   - Use CPU fallback: `--engine cpu`

7. **Build cache corruption**
   - Run `build_torch_clean.cmd` to clean and rebuild
   - Or manually: `cargo clean` then rebuild

### Feature Flag

PyTorch support is optional and controlled by the `torch` feature:

```toml
# In Cargo.toml
[features]
default = []
torch = ["dep:tch"]  # Only include PyTorch when explicitly requested
```

This keeps the base build lightweight while enabling PyTorch when needed.

## Future Enhancements

The current implementation includes infrastructure for:

- **Batch Processing**: `TorchBatchEngine` for population-level GPU evaluation
- **Advanced Optimizers**: Integration with PyTorch optimizers
- **Mixed Precision**: FP16 training for memory efficiency
- **Distributed Training**: Multi-GPU support through PyTorch

## Example Configuration

```json
{
  "name": "PyTorch Trained Model",
  "engine": "Torch",
  "activation": "Tanh",
  "population_size": 512,
  "generations": 200,
  "mutation_rate": 0.02,
  "mutation_strength": 0.05,
  "concurrent": true
}
```

This configuration leverages PyTorch's strengths for large-scale evolution with CUDA acceleration. 