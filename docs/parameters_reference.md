# Learnable Parameters Reference

This document provides a complete breakdown of all learnable parameters in Spectral Neural Networks across different wave modes.

## Overview

Spectral networks learn the **complete wave representation** from data, optimizing:
- **Phase** (wave directions/offsets)
- **Amplitude** (wave strengths)
- **Frequency** (oscillation rates)
- **Position** (spatial localization in Gabor mode)
- **Scale** (wavelet width in Gabor mode)

---

## 2D Mode (`outer_product`)

**Total Parameters**: ~9,446 for a 784→10 layer with 12 waves, 5 harmonics

### Learnable Parameters

| Parameter | Shape | Count | Description |
|-----------|-------|-------|-------------|
| `u` | `[12, 10, 1]` | 120 | Output coordinate vectors (define wave directions in output space) |
| `v` | `[12, 784, 1]` | 9,408 | Input coordinate vectors (define wave directions in input space) |
| `freqs` | `[12, 1, 1]` | 12 | Base frequency per wave |
| `harmonic_freqs` | `[5]` | 5 | Frequency multipliers [1×, 2×, 4×, 8×, 16×] (learnable if `adaptive_freqs=True`) |
| `fourier_coeffs` | `[12, 5]` | 60 | Fourier coefficients (amplitude of each harmonic) |
| `amplitudes` | `[12]` | 12 | Overall amplitude per wave |
| `bias` | `[10]` | 10 | Output bias term |

### What Gets Optimized
```
W = Σ(amplitude_i * Σ(coeff_h * cos(harmonic_h * freq_i * u_i @ v_i)))
```

- **`u`, `v`**: Define the 2D interference pattern geometry (can warp space!)
- **`freqs`**: Controls how fast waves oscillate
- **`harmonic_freqs`**: Octave relationships (if adaptive)
- **`fourier_coeffs`**: Mix of harmonics (like audio EQ)
- **`amplitudes`**: Overall wave strength

---

## 1D Mode (`fourier_series`)

**Total Parameters**: 195 for a 784→10 layer with 12 waves, 5 harmonics

### Learnable Parameters

| Parameter | Shape | Count | Description |
|-----------|-------|-------|-------------|
| `freqs` | `[12]` | 12 | Base frequency per wave (fixed at initialization) |
| `phases` | `[12]` | 12 | **Explicit phase offsets** (NEW in 1D mode!) |
| `harmonic_freqs` | `[5]` | 5 | Frequency multipliers (learnable if adaptive) |
| `fourier_coeffs` | `[12, 5]` | 60 | Fourier coefficients |
| `amplitudes` | `[10, 12]` | 120 | **Per-neuron, per-wave** amplitudes (more expressive!) |
| `bias` | `[10]` | 10 | Output bias term |

### What Gets Optimized
```
W = Σ(amplitude_neuron,wave * Σ(coeff_h * cos(harmonic_h * freq * position + phase)))
```

- **`phases`**: Explicit phase control (horizontal shift of waves)
- **`amplitudes`**: Each output neuron can weight waves independently
- **`fourier_coeffs`**: Harmonic mixing

---

## Gabor Mode

**Total Parameters**: 243 for a 784→10 layer with 12 waves, 5 harmonics

### Learnable Parameters

Includes **all 1D mode parameters** PLUS:

| Parameter | Shape | Count | Description |
|-----------|-------|-------|-------------|
| `centers` | `[12]` | 12 | Spatial localization (where wavelet is centered) |
| `sigmas` | `[12]` | 12 | Scale/width of Gaussian window |

### What Gets Optimized
```
W = Σ(amplitude * gaussian(position, center, sigma) * cos(freq * position + phase))
```

- **`centers`**: Where each wavelet "focuses" in the input space
- **`sigmas`**: How localized vs spread out the wavelet is
- All other parameters from 1D mode

---

## Parameter Groups for Optimization

Different learning rates are applied based on parameter type:

```python
# From trainer.py
LR_PHASE = 1e-3      # u, v, centers (spatial parameters)
LR_AMP = 1e-3        # fourier_coeffs, amplitudes
LR_BIAS = 1e-2       # bias (learns fastest)

# Frequency-aware (advanced)
LR_LOW_FREQ = 1e-3   # Low-frequency harmonics
LR_MID_FREQ = 5e-4   # Mid-frequency harmonics
LR_HIGH_FREQ = 2e-4  # High-frequency harmonics (slowest to combat spectral bias)
```

---

## Key Insights

1.  **2D mode has the most parameters** (~9K) because it learns the full spatial geometry via `u` and `v`.
2.  **1D mode is most parameter-efficient** (195) but still highly expressive via per-neuron amplitudes.
3.  **Gabor mode adds localization** - useful for tasks where features are spatially concentrated.
4.  **All modes learn frequencies** when `adaptive_freqs=True`, allowing the network to discover optimal frequency scales.
5.  **Curriculum learning** can freeze high frequencies initially, then progressively unfreeze them.

---

## Comparison with Standard Layers

| Layer Type | Parameters (784→10) | What's Learned |
|------------|---------------------|----------------|
| **Standard Linear** | 7,850 | Raw weight matrix (no structure) |
| **Spectral 1D** | 195 | Wave frequencies, phases, amplitudes |
| **Spectral 2D** | 9,446 | + Wave geometries (u, v) |
| **Spectral Gabor** | 243 | + Spatial localization (centers, sigmas) |

Spectral layers trade parameter count for **interpretability** and **structured learning** - the network learns a decomposition into physically meaningful wave components.
