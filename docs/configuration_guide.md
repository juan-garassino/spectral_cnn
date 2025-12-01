# Configuration Guide

All Fourier layer configurations are controlled via arguments in `main.py`:

## Core Parameters

```python
NUM_EPOCHS = 3          # Training epochs
NUM_WAVES = 12          # Number of waves per layer
```

## Fourier Configuration

### `NUM_HARMONICS` (default: 3)
Number of Fourier components per wave.
- `3` → wave = c₁·cos(θ) + c₂·cos(2θ) + c₃·cos(4θ)
- `5` → adds c₄·cos(8θ) + c₅·cos(16θ)
- `7` → even richer frequency content

**Trade-off**: More harmonics = more expressiveness but more parameters.

### `ADAPTIVE_FREQS` (default: False)
Whether harmonic frequencies are learnable.
- `False` → Fixed ratios [1×, 2×, 4×, 8×, ...]
- `True` → Network learns optimal frequency ratios

**Use case**: Enable if you suspect non-standard frequency relationships.

### `PER_NEURON_COEFFS` (default: False)
Whether each output neuron has its own Fourier coefficients.
- `False` → All neurons share coefficients `[num_waves, num_harmonics]`
- `True` → Each neuron has unique coefficients `[num_waves, out_dim, num_harmonics]`

**Trade-off**: Massive parameter increase but max flexibility.

### `L1_PENALTY` (default: 0.0)
L1 regularization strength on Fourier coefficients.
- `0.0` → No sparsity constraint
- `0.001` → Mild sparsity (encourages some coefficients → 0)
- `0.01` → Strong sparsity

**Use case**: Encourage network to only use necessary harmonics.

## Example Configurations

### Default (Balanced)
```python
NUM_HARMONICS = 3
ADAPTIVE_FREQS = False
PER_NEURON_COEFFS = False
L1_PENALTY = 0.0
```

### High Fidelity
```python
NUM_HARMONICS = 7          # Rich frequency content
ADAPTIVE_FREQS = True      # Learn optimal ratios
PER_NEURON_COEFFS = False  # Keep params reasonable
L1_PENALTY = 0.001         # Mild sparsity
```

### Maximum Expressiveness
```python
NUM_HARMONICS = 5
ADAPTIVE_FREQS = True
PER_NEURON_COEFFS = True   # ⚠️ 10x more parameters!
L1_PENALTY = 0.01          # Strong sparsity to compensate
```

### Minimal (Fast baseline)
```python
NUM_HARMONICS = 2          # Just fundamental + 1 harmonic
ADAPTIVE_FREQS = False
PER_NEURON_COEFFS = False
L1_PENALTY = 0.0
```

## Parameter Count Impact

For a 784→12 layer with 12 waves:

| Config | Fourier Params | Notes |
|--------|----------------|-------|
| Default (3 harmonics, shared) | 36 | 12 waves × 3 coeffs |
| 7 harmonics | 84 | 12 × 7 |
| Per-neuron (3 harmonics) | 432 | 12 × 12 × 3 |
| Per-neuron (7 harmonics) | 1,008 | 12 × 12 × 7 |
