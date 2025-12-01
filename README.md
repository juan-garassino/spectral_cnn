# Spectral CNN: Weight Parameterization Benchmark

A unified laboratory comparing 7 different mathematical paradigms for neural network weight parameterization on MNIST. This project explores how different basis functions (waves, polynomials, wavelets) can construct weight matrices more efficiently than standard dense layers.

## Overview

The goal of this benchmark is to beat 93% accuracy on MNIST with fewer than 10k parameters (vs ~9.7k for a standard MLP). It compares the following layers:

1.  **Standard**: Conventional `nn.Linear` (Baseline).
2.  **UserWave**: Weights constructed from a sum of cosine waves with learned frequencies and amplitudes.
3.  **Poly**: Weights constructed from polynomial expansions of a low-rank projection.
4.  **Wavelet**: Weights constructed from Morlet-like wavelets (envelope * oscillation).
5.  **Factor**: Low-rank factorization ($U \times V^T$).
6.  **Siren**: Implicit neural representation (SIREN) generating the weight matrix.
7.  **GatedWave**: A hybrid approach using a "Signal" (waves) multiplied by a "Gate" (sigmoid rank-2), optimizing for both structure and sparsity.

## Installation

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

Run the main benchmark script:

```bash
python main.py
```

This will:
1.  Train all 7 models on MNIST for a specified number of epochs.
2.  Print a leaderboard comparing parameters, accuracy, and efficiency.
3.  Generate visualizations in a timestamped folder (e.g., `WeightParam_Benchmark_YYYYMMDD_HHMMSS/`).

### Visualizations

The script generates the following plots:
-   `weight_patterns.png`: Heatmaps of the learned weight matrices (first layer).
-   `frequency_analysis.png`: FFT of the weight matrices to show spectral content.
-   `training_dynamics.png`: Test accuracy curves over epochs.
-   `{Model}_waves.png`: (New) Visualization of the internal basis functions (waves) for supported layers (UserWave, GatedWave).

## Running on Google Colab

1.  **Clone the repository:**
    ```bash
    !git clone https://github.com/YOUR_USERNAME/spectral_cnn.git
    %cd spectral_cnn
    ```

2.  **Install dependencies:**
    ```bash
    !pip install -r requirements.txt
    ```

3.  **Run the benchmark:**
    ```bash
    !python main.py
    ```

4.  **View Results:**
    The results will be saved in a new folder named `WeightParam_Benchmark_...`. You can view the generated images directly in the Colab file browser or display them using python:
    ```python
    from IPython.display import Image
    Image('./WeightParam_Benchmark_.../weight_patterns.png')
    ```

## Project Structure

```
spectral_cnn/
├── main.py                 # Entry point
├── src/
│   ├── models/
│   │   ├── layers.py       # Custom layer implementations (UserWave, Poly, etc.)
│   │   └── networks.py     # UniversalMLP wrapper
│   ├── training/
│   │   └── trainer.py      # Training loop and optimizer logic
│   └── visualization/
│       └── plotter.py      # Plotting functions
└── docs/
    └── theory.md           # Detailed mathematical explanation
```

## Key Results (V14)

The **GatedWave** layer typically achieves the best balance of compression and accuracy, often matching or beating the standard layer with fewer effective parameters by learning a structured "signal" gated by a low-rank mechanism.
