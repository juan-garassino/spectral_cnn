# Understanding Wave Modes: 1D vs 2D

This document explains the fundamental difference between the two wave generation modes in the spectral CNN.

---

## 1D Fourier Series Mode

**Setting:** `WAVE_MODE = "fourier_series"`

### What It Is

Pure sinusoidal waves along the input dimension - like a vibrating string or audio waveform.

```python
weight[j, i] = Σ a_w · cos(2π · freq · i/N + phase)
```

### Visual Representation

```
Position:  0    100   200   300   400   500   600   700
Wave:     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          Smooth sinusoid - looks like: cos(x)
```

Each position along the input gets a value from a smooth sine wave. The same wave pattern is used for all output neurons (scaled by learned amplitudes).

### ASCII Patterns (Shared Coefficients)

```
Neuron 0:  ~~~~~~~~~~~~~~~~~  (wave A scaled by a₀)
Neuron 1:  ~~~~~~~~~~~~~~~~~  (wave A scaled by a₁)
Neuron 2:  ~~~~~~~~~~~~~~~~~  (wave A scaled by a₂)
```

All neurons get the **same horizontal wave**, just with different amplitudes.

### ASCII Patterns (Per-Neuron Coefficients)

```
Neuron 0:  ~~~~~            (mix of waves with coeffs c₀)
Neuron 1:    ~~~~~~         (mix of waves with coeffs c₁)
Neuron 2:  ~~~~             (mix of waves with coeffs c₂)
Neuron 3:     ~~~~~         (mix of waves with coeffs c₃)
```

Each neuron can have completely different wave patterns!

### Analogy: Music Synthesizer 🎵

Think of it like an **audio synthesizer**:
- Each "wave" is a musical note at a specific frequency
- "Harmonics" are overtones (2×, 4×, 8× the base frequency)
- You mix these frequencies to create any sound
- The network learns the perfect "chord" for classification

**Example:**
```
Base:      ∿∿∿∿∿∿∿∿∿∿∿∿  (1 Hz - fundamental)
2nd:       ∿∿∿∿∿∿∿∿∿∿∿∿  (2 Hz - first overtone)
4th:       ∿∿∿∿∿∿∿∿∿∿∿∿  (4 Hz - second overtone)
Combined:  ⩘⩗⩘⩗⩘⩗⩘⩗    (complex waveform)
```

### Characteristics

✅ **Smooth** - Perfect sinusoidal curves  
✅ **Interpretable** - Direct Fourier analysis  
✅ **1D structure** - Only varies along input dimension  
❌ **Limited** - Can't naturally create 2D patterns (unless per-neuron)

---

## 2D Outer Product Mode

**Setting:** `WAVE_MODE = "outer_product"`

### What It Is

An interference field created by the outer product of two learned vectors - like ripples on a pond.

```python
wave = cos(u ⊗ v^T · freq)
```

where:
- `u` [out_dim] - learned vector for output space
- `v` [in_dim] - learned vector for input space

### Visual Representation

```
2D Interference Pattern (like looking down at water):

    ╱╲╱╲╱╲╱╲
   ╱  ╲  ╱  ╲
  ╱    ╲╱    ╲
 ╱╲    ╱╲    ╱╲
   ╲  ╱  ╲  ╱
    ╲╱    ╲╱
     ╱╲    ╱╲
```

Each element `[row, col]` depends on the interaction between `u[row]` and `v[col]`.

### Analogy: Ocean Waves 🌊

Think of it like the **surface of an ocean**:

**Single wave direction:**
```
Direction →
  ∿∿∿∿∿∿∿∿
  ∿∿∿∿∿∿∿∿
  ∿∿∿∿∿∿∿∿
```

**Two waves interfering:**
```
Wave 1 (→) + Wave 2 (↗):
  ╱╲╱╲  ╲╱╲╱
 ╱  ╲╱╲╱  ╲
╱╲  ╱╲╱╲  ╱╲
  ╲╱  ╲╱╲╱
```

The interference creates complex 2D patterns - **moiré effects**, **standing waves**, **anisotropic textures**.

### Another Analogy: Hologram 📻

Like an electromagnetic field or hologram:
- Two coherent wave sources (u and v)
- Their interaction creates an interference pattern
- The pattern encodes complex information
- Can represent any 2D texture with the right combination

### Characteristics

✅ **Expressive** - Can create complex 2D textures naturally  
✅ **Field-like** - Interference, moiré, anisotropic structures  
✅ **Efficient** - Rich patterns with relatively few parameters  
❌ **Less interpretable** - Not pure Fourier components

---

## Side-by-Side Comparison

### 1D Ripple Tank vs 2D Ocean

**1D Fourier (Shared):**
```
Row 0: ~~~~~~~~~~~~~~~~~
Row 1: ~~~~~~~~~~~~~~~~~
Row 2: ~~~~~~~~~~~~~~~~~
Row 3: ~~~~~~~~~~~~~~~~~
```
Like a **ripple tank** - all waves horizontal, only varying left-to-right.

**2D Outer Product:**
```
  ╱╲╱╲  ╲╱╲╱
 ╱  ╲╱╲╱  ╲
╱╲  ╱╲╱╲  ╱╲
  ╲╱  ╲╱╲╱
```
Like the **ocean surface** - waves traveling in different directions, creating interference patterns.

**1D Fourier (Per-Neuron):**
```
Row 0: ~~~~~
Row 1:   ~~~~~~
Row 2: ~~~~
Row 3:    ~~~~~
```
Like a **controlled wave tank** - each row can be different, approximating 2D by stacking 1D patterns.

---

## Comparison Table

| Aspect | 1D Fourier Series | 2D Outer Product |
|--------|------------------|------------------|
| **Dimensionality** | 1D wave | 2D field/texture |
| **Formula** | `cos(freq·i + phase)` | `cos(u·v^T·freq)` |
| **Shape** | Smooth sinusoid | Interference pattern |
| **Analogy** | Music synthesizer 🎵 | Ocean waves 🌊 |
| **Physics** | 1D wave equation | 2D wave interference |
| **Direction** | Horizontal only | Any direction |
| **Structure** | Periodic along input | Anisotropic patterns |
| **Interpretability** | ⭐⭐⭐⭐⭐ High (pure frequencies) | ⭐⭐⭐ Medium (learned fields) |
| **Expressiveness** | ⭐⭐⭐⭐ Good with many waves | ⭐⭐⭐⭐⭐ Very high |
| **Best for** | Sequential data, 1D signals | Spatial data, images |

---

## Which Should You Use?

### Use 1D Fourier Series When:

✅ You want **interpretable** frequency analysis  
✅ You need to understand **which frequencies** the network uses  
✅ Your data has **1D structure** (time series, audio, sequences)  
✅ You want classic **Fourier analysis**  
✅ You're willing to use `PER_NEURON_COEFFS=True` for more flexibility

**Example domains:**
- Audio processing
- Time series prediction
- Signal analysis
- Any 1D sequential data

### Use 2D Outer Product When:

✅ You want **maximum expressiveness**  
✅ Your data has **2D spatial structure** (images, textures)  
✅ You're willing to trade interpretability for power  
✅ You want **efficient parameter usage**  
✅ You need **directional patterns** and **interference effects**

**Example domains:**
- Image classification (like MNIST)
- Texture synthesis
- Spatial pattern recognition
- Any 2D grid data

---

## Mathematical Deep Dive

### 1D Fourier Series

Each weight element is a **sum of sinusoids**:

```
weight[j, i] = Σ_{w=1}^{NUM_WAVES} a_{j,w} · Σ_{h=1}^{NUM_HARMONICS} c_{w,h} · cos(2π·f_w·k_h·(i/N) + φ_w)
```

**Learnable parameters:**
- `a_{j,w}` - amplitude for neuron j, wave w
- `c_{w,h}` - coefficient for wave w, harmonic h
- `f_w` - base frequency for wave w
- `k_h` - harmonic multiplier for harmonic h (learnable if `ADAPTIVE_FREQS=True`)
- `φ_w` - phase for wave w

**Total components per layer:** `NUM_WAVES × NUM_HARMONICS`

### 2D Outer Product

Each wave is a **2D interference pattern**:

```
wave = cos(u ⊗ v^T · freq) = cos(u[j] · v[i]^T · freq)
```

Then mixed with Fourier harmonics:

```
weight[j, i] = Σ_{w=1}^{NUM_WAVES} a_w · Σ_{h=1}^{NUM_HARMONICS} c_{w,h} · cos(k_h · θ_w[j,i])
```

where `θ_w[j,i] = u_w[j] · v_w[i] · freq_w`

**Learnable parameters:**
- `u_w` - output space vector for wave w
- `v_w` - input space vector for wave w
- `freq_w` - frequency for wave w
- `c_{w,h}` - harmonic coefficients
- `a_w` - wave amplitudes

**Creates:** Natural 2D structure through outer product!

---

## Performance Expectations

Both modes can achieve similar final accuracy with enough capacity, but they have different **learning dynamics** and **inductive biases**.

### 1D Mode
- Faster convergence on 1D-structured data
- May need more waves/harmonics for 2D patterns
- Very interpretable results

### 2D Mode
- Faster convergence on spatial data
- More parameter-efficient for images
- Richer visual patterns

**On MNIST (2D images):** 2D outer product typically performs better!  
**On sequences/audio:** 1D Fourier series is more natural!

---

## Combining the Best of Both

You can even mix both! Use:
- **1D mode** with `PER_NEURON_COEFFS=True` for flexibility
- **High `NUM_WAVES` and `NUM_HARMONICS`** for capacity
- **L1 sparsity** to prevent overfitting

This gives you interpretable 1D frequencies that can approximate 2D patterns!

---

## Summary

Think of it this way:

**1D Fourier = Playing notes on a piano** 🎹  
Each key is a frequency, you combine them to make music.

**2D Outer Product = Waves on the ocean** 🌊  
Waves interfere from different directions, creating complex surfaces.

Both are beautiful manifestations of wave phenomena - choose based on your data structure and interpretability needs! 🌊🎵
