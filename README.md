# Spectral CNN: Holographic General Intelligence 🌊🧠

**Wave-based neural networks that learn continuous field representations instead of discrete weights.**

A comprehensive framework exploring how AI might generalize better by working with spectral (wave) representations, inspired by real-world phenomena like interference, superposition, and holography. This project implements spectral layers for 1D, 2D, and 3D data, with advanced features like frequency curriculum learning and progressive unfreezing.

---

## 🎯 Core Hypothesis: The Holographic Generalization Principle

> *"AI would generalize better if it worked with waves... interferences and superpositions, holographic projections... more like the real world."*

Current AI uses **discrete atomism** (grids of independent pixels/weights). The real world operates on **continuous spectra** (light waves, sound frequencies, quantum interference). Spectral networks bridge this gap by learning wave-based representations.

**Key Insight**: In a hologram, every part contains the whole. Cutting a photograph loses information; cutting a hologram loses resolution, not content. Spectral networks achieve similar **distributed representations** where information is spread across the frequency spectrum.

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy rich

# Run basic benchmark
python main.py

# Explore inner wave structure
python demo/inspect_inner_waves.py

# Test curriculum learning
python demo/demo_curriculum_learning.py
```

---

## 🌊 What Makes This Different?

### Traditional Neural Networks
```
Weight Matrix = Random initialization → Gradient descent → ???
```

### Spectral Neural Networks
```
Weight Matrix = Σ(Amplitude × Wave) where each wave has:
  • Frequency (oscillation rate)
  • Phase (spatial offset)
  • Amplitude (strength)
  • Harmonics (octave relationships)
```

**Result**: Interpretable, structured learning with physically meaningful decomposition.

---

## 📊 Layer Types

| Layer | Dimensions | Parameters (784→10) | Use Case |
|-------|-----------|---------------------|----------|
| **SpectralConv1d** | 1D | ~195 | Time series, audio, sequences |
| **SpectralConv2d** | 2D | ~9,446 | Images, surfaces |
| **SpectralConv3d** | 3D | ~50,000 | MRI scans, video, volumetric data |
| **UserWaveLinear** | Any | Configurable | Fully-connected layers with wave structure |

All layers support:
- **Multiple wave modes**: `outer_product` (2D), `fourier_series` (1D), `gabor` (localized wavelets)
- **Adaptive frequencies**: Network learns optimal frequency scales
- **Harmonic stacking**: Multi-resolution decomposition (1×, 2×, 4×, 8× base frequency)
- **Curriculum learning**: Progressive frequency unfreezing

---

## 🎓 Advanced Features

### 1. Frequency Curriculum Learning

Combat spectral bias by starting with low frequencies and progressively adding detail:

```python
from src.models.networks import UniversalMLP

model = UniversalMLP("UserWave", num_waves=12, num_harmonics=5)

# Start with only low frequencies
model.fc1.freeze_high_frequencies(threshold=0.2)

# During training: progressively unfreeze
for epoch in range(num_epochs):
    model.fc1.progressive_unfreeze_schedule(epoch, num_epochs, strategy='linear')
```

**Benefits**:
- Faster convergence
- Better use of high-frequency components
- Smoother optimization landscape

### 2. Frequency-Aware Optimization

Different learning rates for different frequency bands:

```python
from src.training.trainer import get_optimizer

# Low freq = 1e-3, Mid freq = 5e-4, High freq = 2e-4
optimizer = get_optimizer(model, mode="UserWave", frequency_aware=True)
```

### 3. Multi-Dimensional Wave Processing

Universal wave computing across all dimensions:

```python
# 1D: Time series
conv1d = SpectralConv1d(in_channels=1, out_channels=32, kernel_size=7)

# 2D: Images
conv2d = SpectralConv2d(in_channels=3, out_channels=64, kernel_size=5)

# 3D: Volumetric data
conv3d = SpectralConv3d(in_channels=1, out_channels=32, kernel_size=3)
```

---

## 📈 Visualization Dashboard

The framework generates comprehensive analysis dashboards:

```bash
python tests/test_dashboard.py
```

**Output** (`results/comprehensive_dashboard.png`):
- **Weight Patterns**: Spatial domain visualization
- **Frequency Analysis**: 2D FFT showing spectral content
- **Learning Curves**: Training dynamics
- **Performance Table**: Accuracy, parameters, inference speed

---

## 🔬 Learnable Parameters

### What the Network Optimizes

All wave-defining parameters are learned from data:

- ✅ **Phase** (u, v in 2D; explicit phases in 1D)
- ✅ **Amplitude** (fourier_coeffs + amplitudes)
- ✅ **Frequency** (freqs + harmonic_freqs if adaptive)
- ✅ **Position** (centers in Gabor mode)
- ✅ **Scale** (sigmas in Gabor mode)
- ✅ **Bias** (output bias terms)

See [docs/parameters_reference.md](docs/parameters_reference.md) for complete breakdown.

---

## 📚 Documentation

- **[Spectral Manifesto](docs/spectral_manifesto.md)**: Vision of Holographic General Intelligence
- **[Theoretical Extensions](docs/theoretical_extensions.md)**: Research directions and hypotheses
- **[Parameters Reference](docs/parameters_reference.md)**: Complete parameter breakdown for all modes
- **[Theory](docs/theory.md)**: Mathematical foundations

---

## 🗂️ Project Structure

```
spectral_cnn/
├── main.py                          # Main benchmark runner
├── src/
│   ├── models/
│   │   ├── layers.py                # Spectral layers (1D/2D/3D, all wave modes)
│   │   └── networks.py              # SpectralCNN, UniversalMLP
│   ├── training/
│   │   └── trainer.py               # Training loop, frequency-aware optimizer
│   └── visualization/
│       └── plotter.py               # Comprehensive dashboards
├── demo/
│   ├── inspect_inner_waves.py       # Visualize individual wave channels
│   ├── demo_curriculum_learning.py  # Progressive frequency unfreezing demo
│   └── analyze_parameters.py        # Parameter analysis tool
├── tests/
│   └── test_dashboard.py            # Dashboard generation test
└── docs/
    ├── spectral_manifesto.md        # Vision and philosophy
    ├── theoretical_extensions.md    # Research ideas
    ├── parameters_reference.md      # Complete parameter guide
    └── theory.md                    # Mathematical foundations
```

---

## 🚀 Key Innovations

1. **Universal Wave Computing**: Same math works for 1D audio, 2D images, 3D volumes
2. **Frequency Curriculum**: Start simple (low freq), add complexity (high freq) during training
3. **Holographic Representation**: Information distributed across spectrum, not localized
4. **4D Spacetime Prediction**: Treating time as another dimension for forecasting
5. **Infinite Resolution**: Query network at any coordinate for resolution-independent output

---

## 🎯 Performance

On MNIST (baseline):
- **Standard MLP**: 93% accuracy, 7,850 parameters
- **Spectral 1D**: 90%+ accuracy, 195 parameters (40× compression!)
- **Spectral 2D**: 93%+ accuracy, 9,446 parameters (structured learning)

With curriculum learning:
- **+2-5% accuracy** improvement
- **Faster convergence** (fewer epochs to target accuracy)
- **Better high-frequency utilization**

---

## 🔮 Future Directions

See [docs/theoretical_extensions.md](docs/theoretical_extensions.md) for:
- Complex-valued networks (true phase representation)
- Continuous-domain processing (output functions, not tensors)
- Fractal architectures (self-similar processing)
- Quantum-inspired interference learning

---

## 🙏 Acknowledgments

Inspired by:
- Fourier Neural Operators (Li et al.)
- SIREN (Sitzmann et al.)
- Implicit Neural Representations
- Holographic principles in physics
- Progressive GANs (Karras et al.)

---

## 📄 License

MIT License - See LICENSE file for details

---

**"We are not just training networks; we are composing symphonies of intelligence."** 🎼🌊
