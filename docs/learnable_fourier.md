# Learnable Fourier Components - Architecture Summary

## What We Optimized

Previously, the Fourier coefficients were **hardcoded**:
```python
wave = 1.0·cos(θ) + 0.5·cos(2θ) + 0.25·cos(4θ)
```

Now, each wave has **3 learnable Fourier coefficients** `[c₁, c₂, c₃]`:
```python
wave = c₁·cos(θ) + c₂·cos(2θ) + c₃·cos(4θ)
```

## Optimization Hierarchy

```
Layer
  ├─ Wave 0
  │   ├─ Component 1: c₁·cos(θ)      [OPTIMIZED]
  │   ├─ Component 2: c₂·cos(2θ)     [OPTIMIZED]
  │   ├─ Component 3: c₃·cos(4θ)     [OPTIMIZED]
  │   └─ Combined: c₁·cos(θ) + c₂·cos(2θ) + c₃·cos(4θ)
  │
  ├─ Wave 1
  │   ├─ Component 1: c₁·cos(θ)      [OPTIMIZED]
  │   ├─ Component 2: c₂·cos(2θ)     [OPTIMIZED]
  │   └─ ...
  │
  └─ ... (12 waves total)
      
Final Weight Matrix = Σ(amplitude[i] × wave[i])
```

## Parameters Being Optimized

For UserWave and GatedWave layers:

1. **Phase Parameters** (`u`, `v`, `freqs`): Control the spatial structure (θ)
2. **Fourier Coefficients** (`fourier_coeffs[num_waves, 3]`): Control harmonic mix
3. **Amplitudes** (`amplitudes`): Overall wave strength
4. **Bias**: Output bias

## Learning Rates

Different parameter groups use different learning rates:
- **Phase** (u, v): `LR_PHASE = 0.0050` - Fast frequency tuning
- **Amplitude/Fourier**: `LR_AMP = 0.0020` - Moderate signal growth
- **Bias**: `LR_BIAS = 0.0010` - Slow baseline adjustment

## What the Network Learns

The network now automatically discovers:
- **Which harmonics matter** (c₁, c₂, c₃ values)
- **How to mix them** (ratios between components)
- **Frequency content** (via θ and freqs)

## Visualizations Show

1. **Individual Fourier Components**: The 3 learned harmonics
2. **Combined Wave**: How components superpose
3. **Learned Coefficients Display**: Exact values of c₁, c₂, c₃
4. **Frequency Spectrum**: Bar chart of learned amplitudes

This is now a **true spectral learning system** where the Fourier basis itself is optimized!
