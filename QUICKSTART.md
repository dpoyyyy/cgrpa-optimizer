# CGRPA Optimizer - Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dpoyyyy/cgrpa-optimizer.git
cd cgrpa-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the MNIST Example

```bash
python train_mvp.py
```

Expected output:
```
Using device: cpu/gpu
Loading MNIST dataset...
Model architecture:
SimpleMLP(...)
Optimizer: CGRPAOptimizer (lr=0.005)

Starting training...
Epoch [1/2], Batch [100/469], Loss: 0.4523
Epoch [1/2], Batch [200/469], Loss: 0.2134
...
Training completed successfully!
Training Accuracy: 95.23%
```

## Basic Usage

### Minimal Example

```python
from structure import CGRPAOptimizer

# Create optimizer
optimizer = CGRPAOptimizer(model.parameters(), lr=1e-3)

# Training loop
for batch in dataloader:
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(batch.data), batch.target)
        loss.backward(retain_graph=True)  # Important!
        return loss
    
    loss = optimizer.step(closure)
```

### Important Notes

1. **Closure is required**: Unlike SGD/Adam, CGRPA needs a closure function
2. **Retain graph**: Must use `retain_graph=True` in backward pass
3. **Gradient clipping**: Highly recommended for stability

## Hyperparameter Tuning

Start with these ranges:

| Parameter | Range | Default |
|-----------|-------|---------|
| lr | 1e-4 to 5e-3 | 1e-3 |
| tau_max | 2.0 to 20.0 | 5.0-10.0 |
| betas[0] | 0.85 to 0.95 | 0.9 |

## Common Issues

### NaN Loss
- Reduce learning rate
- Lower tau_max
- Ensure gradient clipping is enabled

### Slow Convergence
- Increase learning rate slightly
- Increase tau_max (more aggressive Newton steps)

### Memory Issues
- Reduce batch size
- The optimizer stores per-parameter state but no large matrices

## Next Steps

- Read the full [README.md] and [ALGORITHM.md] for detailed algorithm description
- Check [CONTRIBUTING.md] to contribute
- Open an issue for questions or bug reports
