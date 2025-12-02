# Universal Approximation with Fourier Series

## Theory

With the 1D Fourier series mode, you can approximate **any continuous periodic function** to arbitrary precision by increasing `NUM_WAVES` and `NUM_HARMONICS`.

## Flexibility Features

All parameters are **fully learnable**:

### 1. Base Frequencies (`freqs`)
- **What**: The fundamental frequency of each wave
- **Learnable**: Yes! Initialized randomly in [0.5, 10.5]
- **Effect**: Network discovers optimal frequency distribution

### 2. Harmonic Multipliers (`harmonic_freqs`)
- **What**: Frequency ratios for each harmonic component
- **Learnable**: If `ADAPTIVE_FREQS=True` (otherwise fixed [1×, 2×, 4×, 8×, ...])
- **Effect**: Non-integer harmonics (e.g., 1.3×, 2.7×) for richer textures

### 3. Phase Shifts (`phases`)
- **What**: Phase offset for each wave
- **Learnable**: Yes! Initialized randomly in [0, 2π]
- **Effect**: Allows waves to align optimally

### 4. Fourier Coefficients (`fourier_coeffs`)
- **What**: Amplitude of each harmonic component
- **Learnable**: Yes! Network optimizes the harmonic mix
- **Effect**: Determines which frequencies are important

### 5. Wave Amplitudes (`amplitudes`)
- **What**: Overall strength of each wave per output neuron
- **Learnable**: Yes! Shape [out_dim, num_waves]
- **Effect**: Each neuron picks which waves it needs

## Universal Approximation

**Formula for each output neuron j:**
```
weight[j, i] = Σ_{w=1}^{NUM_WAVES} a_{j,w} · Σ_{h=1}^{NUM_HARMONICS} c_{w,h} · cos(2π · f_w · k_h · (i/N) + φ_w)
```

Where:
- `a_{j,w}` = amplitude for neuron j, wave w (learnable)
- `c_{w,h}` = coefficient for wave w, harmonic h (learnable)
- `f_w` = base frequency for wave w (learnable)
- `k_h` = harmonic multiplier for harmonic h (learnable if ADAPTIVE_FREQS=True)
- `φ_w` = phase for wave w (learnable)

**Result**: With enough waves and harmonics, this can approximate ANY weight pattern!

## Recommended Settings for Maximum Flexibility

```python
# In main.py
NUM_WAVES = 20              # More waves = more basis functions
NUM_HARMONICS = 7           # More harmonics = richer frequency content
ADAPTIVE_FREQS = True       # Learn optimal harmonic ratios
PER_NEURON_COEFFS = True    # Each neuron optimizes its own mix
L1_PENALTY = 0.001          # Mild sparsity to prevent overfitting
WAVE_MODE = "fourier_series"  # Use 1D smooth sinusoids
```

## How It Works

1. **With 1 wave, 1 harmonic**: Single sine wave (very limited)
2. **With 3 waves, 3 harmonics**: 9 frequency components (basic)
3. **With 12 waves, 5 harmonics**: 60 frequency components (good)
4. **With 20 waves, 7 harmonics**: 140 frequency components (excellent!)

Each additional wave or harmonic adds more **degrees of freedom**, allowing the network to fit increasingly complex patterns.

## Comparison

| Config | Frequency Components | Can Approximate |
|--------|---------------------|-----------------|
| 3 waves × 3 harmonics | 9 | Simple curves |
| 12 waves × 5 harmonics | 60 | Most patterns |
| 20 waves × 7 harmonics | 140 | Almost anything! |
| 50 waves × 10 harmonics | 500 | Universal! |

The network **automatically discovers** which frequencies are needed through gradient descent! 🌊
