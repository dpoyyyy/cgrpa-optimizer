# CGRPA Optimizer

**Curvature-Gated Recursive Parameter Anchoring** - A novel second-order PyTorch optimizer that combines curvature estimation, variance-based reliability gating, and recursive anchor dynamics for stable and efficient optimization.

## Overview

CGRPA is a second-order optimization method designed to leverage curvature information while maintaining numerical stability. Unlike traditional second-order methods that require expensive Hessian computations or matrix inversions, CGRPA uses:

- **Hutchinson Diagonal Estimation**: Efficient stochastic estimation of Hessian diagonal via randomized trace estimation
- **Variance-Gated Reliability (VGR)**: Adaptive gating mechanism that modulates trust in curvature estimates based on their variance
- **Regularized Projected Anchor (RPA)**: Coupled anchor-parameter dynamics that prevent divergence in high-curvature regions
- **Trust Region Scaling**: Bounded Newton-like steps that gracefully degrade to SGD under uncertainty

## Features

-  **Second-order optimization** without full Hessian computation
-  **Automatic reliability detection** via variance monitoring
-  **Numerically stable** with extensive safety guards
-  **PyTorch native** implementation
-  **Drop-in replacement** for standard optimizers like Adam or SGD
-  **GPU compatible**

## Installation

### Requirements

```
torch >= 1.9.0
torchvision (for examples)
```

### Setup

Clone the repository and ensure all modules are in the same directory:

```bash
git clone https://github.com/dpoyyyy/cgrpa-optimizer.git
cd cgrpa-optimizer
```

## Quick Start

```python
import torch
import torch.nn as nn
from structure import CGRPAOptimizer

# Define your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Initialize CGRPA optimizer
optimizer = CGRPAOptimizer(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    tau_max=10.0,
    weight_decay=0.0
)

# Training loop
for data, target in dataloader:
    # Define closure (required for second-order methods)
    def closure():
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)  # Important: retain graph for curvature estimation
        return loss
    
    # Optimizer step
    loss = optimizer.step(closure)
```

## Algorithm Components

### 1. Hutchinson Curvature Sensor

Estimates the diagonal of the Hessian matrix using a Monte Carlo approach:

```
diag(H) ≈ v ⊙ Hv
```

where `v` is a Rademacher random vector ({-1, +1}) and `Hv` is computed via automatic differentiation.

### 2. Variance-Gated Reliability (VGR)

Maintains exponential moving averages of curvature statistics:

```
m_t = β·m_{t-1} + (1-β)·c_t
s_t = β·s_{t-1} + (1-β)·c_t²
σ²_t = s_t - m_t²
g_t = 1/(1 + σ²_t)
```

The gate `g_t` ranges from 0 (unreliable) to 1 (reliable).

### 3. Direction Modifier

Blends SGD and Newton-like directions based on reliability:

```
κ_t = |m_t| + ε
τ_t = min(1/κ_t, τ_max)
λ_t = (1 - g_t) + g_t·τ_t
d_t = λ_t·∇L
```

### 4. Regularized Projected Anchor (RPA)

Updates parameters via coupled anchor dynamics:

```
ω_t = 1/(1 + κ_t)
a_t = a_{t-1} - α·ω_t·d_t
θ_t = (1 - ω_t)·θ_{t-1} + ω_t·a_t
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate (α) |
| `betas` | (0.9, 0.999) | EMA coefficients for VGR |
| `eps` | 1e-8 | Numerical stability term |
| `tau_max` | 5.0-10.0 | Maximum trust region scaling |
| `weight_decay` | 0.0 | L2 regularization coefficient |

## Examples

### MNIST Classification

Run the included example:

```bash
python train_mvp.py
```

This trains a simple MLP on MNIST and demonstrates the optimizer's stability and convergence properties.

### Custom Models

```python
# For ResNets, Transformers, or other architectures
optimizer = CGRPAOptimizer(
    model.parameters(),
    lr=5e-4,           # Often works well with smaller lr than Adam
    tau_max=5.0,       # Conservative trust region for deep networks
    weight_decay=1e-4  # Standard L2 regularization
)
```

## Performance Characteristics

### Advantages
- **Curvature awareness**: Adapts step size based on local geometry
- **Automatic fallback**: Reduces to SGD in flat or uncertain regions
- **Stability**: Extensive clamping and NaN handling prevents training collapse
- **Memory efficient**: Only stores per-parameter scalars, no large matrices

### Considerations
- **Computational cost**: ~2x gradient computation cost (Hutchinson requires second-order derivatives)
- **Closure required**: Must provide a closure function for curvature estimation
- **Graph retention**: Requires `retain_graph=True` in backward pass

## Best Practices

1. **Use gradient clipping**: Recommended for stability
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Start with conservative hyperparameters**:
   - lr: 1e-3 to 5e-3
   - tau_max: 5.0 to 10.0

3. **Monitor for NaN/Inf**: The optimizer includes safety checks but monitoring is recommended

4. **Adjust tau_max for your task**:
   - Lower (2.0-5.0) for very deep networks
   - Higher (10.0-20.0) for shallow networks or convex problems

## Architecture

```
CGRPAOptimizer (structure.py)
├── Hutchinson_Curvature_Sensor.py    # Hessian diagonal estimation
├── VGR.py                             # Variance-gated reliability
├── Direction_Modifier.py              # Adaptive direction computation
└── RPA_Core.py                        # Anchor-based parameter updates
```

## Citation

If you use this optimizer in your research, please cite:

```bibtex
@misc{cgrpa2026,
  title={CGRPA: Curvature-Gated Recursive Parameter Anchoring},
  author={Danial farshbaf},
  year={2026},
  url={https://github.com/dpoyyyy/cgrpa-optimizer}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This optimizer was developed through iterative design and mathematical analysis to ensure stable second-order optimization in deep learning contexts.

## Contact

For questions or issues, please open an issue on GitHub or contact [Danielfarshbaf@gmail.com].
