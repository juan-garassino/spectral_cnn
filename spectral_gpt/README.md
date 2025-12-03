# 🌊 Unified Spectral GPT

**One Architecture to Rule Them All.**

This package unifies the **Spectral Transformer** (Wave-based Attention) and the **Spectral Curriculum GPT** (FFT-based Mixing) into a single, highly configurable framework. It also introduces "Physics-First" strategies that treat the model as a quantum field simulator rather than a statistical machine.

## 🚀 Quick Start

### 1. Run the Benchmark ("Battle Royale")
Compare Classic Transformer, Spectral Transformer, FFT Mixer, and the new Physics Spectral model.
```bash
python benchmark.py
```
*Outputs results to `benchmark_results/` (plots, metrics, models).*

### 2. Train a Specific Model
**The "Transformer Killer" (FFT Mixing + Standard Weights)**
```bash
python train.py --layer-type fft --weight-type standard --exp-name fft_run
```

**The "Spectral Transformer" (Attention + Wave Weights)**
```bash
python train.py --layer-type attention --weight-type wave --exp-name wave_run
```

**The "Physics Spectral" (FFT + Wave Weights + Physics Init)**
```bash
python train.py --layer-type fft --weight-type wave \
    --init-mode holographic --activation-type modulate \
    --use-hamiltonian --use-collapse \
    --exp-name physics_run
```

---

## 🧠 Architectures

### 1. Layer Type (`--layer-type`)
*   **`fft`**: Replaces Self-Attention with a Global FFT Mixing layer.
    *   **Complexity**: $O(N \log N)$ (vs $O(N^2)$ for Attention).
    *   **Mechanism**: Transforms sequence to frequency domain, applies a learnable filter, and transforms back.
*   **`attention`**: Standard Multi-Head Self-Attention.
    *   **Complexity**: $O(N^2)$.
    *   **Mechanism**: Query-Key-Value interactions.

### 2. Weight Type (`--weight-type`)
*   **`standard`**: Uses standard `nn.Linear` (dense matrices).
*   **`wave`**: Uses `UserWaveLinear`. Weights are continuous functions parameterized by sums of sines (Fourier Series).
    *   **Benefit**: Infinite resolution, inductive bias for smoothness, parameter efficiency.

---

## ⚛️ Physics-First Strategies

We model the network as a physical system obeying wave mechanics.

### Initialization (`--init-mode`)
*   **`standard`**: Random Gaussian noise.
*   **`dft`**: Weights initialized as perfect Discrete Fourier Transform matrices (Harmonic Basis).
*   **`holographic`**: Weights follow a $1/f$ (Pink Noise) power law, matching natural signal statistics.
*   **`standing_wave`**: Embeddings initialized as Gaussian wave packets (Gabor wavelets).

### Dynamics
*   **Hamiltonian Descent (`--use-hamiltonian`)**: Enforces **Energy Conservation** (Unitary Constraints). The model must trade off energy between frequencies; it cannot just explode.
*   **Wave Function Collapse (`--use-collapse`)**: A sparsity penalty (Sadar Effect) that forces the "probability cloud" of waves to crystallize into a sharp structure.

### Activation (`--activation-type`)
*   **`gelu`**: Standard deep learning activation.
*   **`bilinear`**: SwiGLU (State-of-the-art LLM standard).
*   **`modulate`**: **$x \cdot \cos(x)$**. A "Mixer" non-linearity that creates sideband frequencies ($f_1 \pm f_2$) instead of just rectifying signals.

---

## 📊 Benchmark Suite

The `benchmark.py` script runs a comprehensive comparison:
1.  **Classic Transformer**: Attention + Standard Weights.
2.  **Spectral Transformer**: Attention + Wave Weights.
3.  **FFT Mixer (GFNet)**: FFT + Standard Weights.
4.  **Full Spectral**: FFT + Wave Weights.
5.  **Physics Spectral**: FFT + Wave Weights + Holographic Init + Hamiltonian + Collapse + Modulation.

**Metrics:**
*   **Speed**: Tokens/sec.
*   **Memory**: Peak VRAM.
*   **Quality**: Perplexity & Validation Loss.
*   **Scaling**: Includes a "Stress Test" with context length $N=2048$ to demonstrate FFT's efficiency.
