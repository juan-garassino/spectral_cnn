# Wave Modes: Outer Product vs. Fourier Series

## Two Approaches to Spectral Weights

The spectral layers now support two different wave generation modes, selectable via the `wave_mode` parameter.

### Mode 1: Outer Product (Default - Original)

**Config:** `WAVE_MODE = "outer_product"`

**How it works:**
```python
wave = cos(u · v^T · freq)
```
- Creates **2D interference patterns** via outer product
- `u` and `v` are learned vectors
- Results in **textured, 2D patterns** (not smooth sinusoids)

**Why 1D slices aren't smooth:**
The outer product creates a 2D pattern where each element `[i,j]` depends on the interaction between `u[i]` and `v[j]`. When you slice through it, you're cutting through this 2D interference pattern, which creates irregular shapes.

**Advantages:**
- More expressive (can learn arbitrary 2D patterns)
- Proven to work well in the current benchmark

**Disadvantages:**
- Harder to interpret
- Not true Fourier basis functions

---

### Mode 2: Fourier Series (**NEW!**)

**Config:** `WAVE_MODE = "fourier_series"`

**How it works:**
```python
wave[i] = Σ c_k · cos(2π · freq_k · (i/N) + phase_k)
```
- Creates **1D smooth sinusoidal waves** along input dimension
- True Fourier series decomposition
- Each output neuron mixes these waves with learned amplitudes

**Why they ARE smooth:**
These are classical Fourier basis functions - pure cosine waves at different frequencies. The 1D slice IS the wave itself!

**Advantages:**
- **Interpretable!** You see exactly what you expect: smooth sine waves
- Classic signal processing approach
- Each wave is a pure frequency component

**Disadvantages:**
- May be less expressive than 2D patterns
- More parameters in some configurations

---

## Switching Between Modes

In `main.py`:
```python
WAVE_MODE = "outer_product"    # Original 2D patterns
# or
WAVE_MODE = "fourier_series"   # Smooth 1D sinusoids
```

Both modes support all the same configuration options:
- `num_harmonics` - How many frequency components per wave
- `adaptive_freqs` - Learnable frequency ratios
- `per_neuron_coeffs` - Per-neuron vs shared coefficients
- `l1_penalty` - Sparsity regularization

## What to Expect in Visualizations

**Outer Product Mode:**
- Waves look like 2D textures/patterns
- 1D slices show irregular curves
- Heatmaps show interference patterns

**Fourier Series Mode:**
- Waves are smooth sinusoids
- 1D slices show perfect cosine curves  
- Heatmaps show the same wave repeated across neurons

## Which to Use?

- **Outer Product:** If you want maximum expressiveness and don't need interpretability
- **Fourier Series:** If you want classic Fourier analysis with interpretable smooth waves

Try both and compare performance! 🌊
