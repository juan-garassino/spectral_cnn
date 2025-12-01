# Theory: Spectral Weight Parameterization

This document details the mathematical foundations of the custom layers implemented in this project. The core idea is to replace the free parameters of a standard weight matrix $W \in \mathbb{R}^{out \times in}$ with a function $f(\theta)$ parameterized by fewer variables.

## 1. UserWave Layer

**Concept:** The weight matrix is constructed as a sum of cosine waves. This assumes that the features in the data have some spectral structure that can be captured by frequency components.

**Formulation:**
Let $u \in \mathbb{R}^{K \times out \times 1}$ and $v \in \mathbb{R}^{K \times in \times 1}$ be learnable vectors for $K$ waves.
We compute a base phase $\theta_{base} = u \cdot v^T$.
The final weight $W$ is a sum over $K$ waves:

$$ W_{ij} = \sum_{k=1}^{K} A_k \cdot \text{Wave}(\theta_{ijk} \cdot \omega_k) $$

Where:
- $\omega_k$ are learnable frequencies (initialized exponentially).
- $A_k$ are learnable amplitudes.
- $\text{Wave}(x) = \cos(x) + 0.5\cos(2x) + 0.25\cos(4x)$ (A rich harmonic waveform).

**Intuition:** This is similar to a Fourier series approximation of the weight matrix, but the "grid" is learned via the low-rank outer products $u \cdot v^T$.

## 2. Poly Layer

**Concept:** Instead of trigonometric functions, this layer uses polynomial expansion to construct the weights.

**Formulation:**
$$ \theta = u \cdot v^T $$
$$ t = \tanh(\theta) $$
$$ W = \sum_{k=1}^{K} c_k \cdot t^{k+1} $$

Where $c_k$ are learnable coefficients.

**Intuition:** This is a Taylor series-like expansion. The $\tanh$ bounds the input to $[-1, 1]$ to ensure stability of the polynomial powers.

## 3. Wavelet Layer

**Concept:** Uses Morlet-like wavelets (a sine wave multiplied by a Gaussian envelope) to localize features in the weight matrix.

**Formulation:**
$$ \theta = u \cdot v^T $$
$$ W = \sum_{k=1}^{K} A_k \cdot e^{-\theta_k^2} \cdot \cos(5 \theta_k) $$

**Intuition:** Wavelets are excellent at capturing local discontinuities and features, unlike global Fourier waves. This allows the layer to be "sparse" in the frequency domain but spatially localized.

## 4. Factor Layer (Low-Rank)

**Concept:** Standard Matrix Factorization.

**Formulation:**
$$ W = U \cdot V^T $$
Where $U \in \mathbb{R}^{out \times r}$ and $V \in \mathbb{R}^{in \times r}$ with rank $r \ll \min(out, in)$.

**Intuition:** Assumes the weight matrix is low-rank. This is the simplest form of compression.

## 5. Siren Layer

**Concept:** Uses a Coordinate-Based Neural Network (MLP) to generate the weight matrix. This is also known as a Hypernetwork or Implicit Neural Representation.

**Formulation:**
We create a grid of coordinates $(y, x)$ for every element in $W$.
These coordinates are passed through a small MLP with Sine activations (SIREN):
$$ W_{ij} = \text{MLP}(\sin(\omega [i, j])) $$

**Intuition:** SIRENs are known for being able to represent high-frequency details (like images or 3D shapes). Here, we treat the weight matrix as an "image" to be generated.

## 6. GatedWave Layer (The "Optimized Duel")

**Concept:** A hybrid approach that combines the spectral power of **UserWave** with a spatial **Gating** mechanism.

**Formulation:**
1.  **Signal**: Generated exactly like the UserWave layer (sum of harmonics).
    $$ S = \text{UserWave}(\theta) $$
2.  **Gate**: A low-rank sigmoid mask.
    $$ G = \sigma(U_g V_g^T + b_g) $$
3.  **Output**:
    $$ W = S \odot G $$

**Intuition:**
-   The **Signal** captures the global texture/frequency patterns of the weights (e.g., "detect vertical edges").
-   The **Gate** determines *where* these patterns are active (e.g., "only in the center of the image").
-   This separation of concerns allows for high compression while maintaining the ability to be spatially precise.
