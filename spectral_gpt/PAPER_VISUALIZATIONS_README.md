# Paper Visualization Generators

This module provides specialized visualization generators for creating publication-quality figures for academic papers about Spectral GPT.

## Overview

The module consists of three main classes:

1. **ArchitectureDiagramGenerator**: Creates architecture diagrams comparing Standard Transformer and Spectral GPT
2. **LossLandscapeVisualizer**: Generates loss landscape visualizations with optimization trajectories
3. **FrequencySpectrumVisualizer**: Creates frequency spectrum and wave property visualizations

## Installation

The visualization generators require the following dependencies:

```bash
pip install numpy matplotlib torch
```

## Quick Start

```python
from spectral_gpt.paper_visualizations import (
    ArchitectureDiagramGenerator,
    LossLandscapeVisualizer,
    FrequencySpectrumVisualizer
)

# Create output directory
output_dir = "experiments/paper/figures"

# Generate architecture diagrams
arch_gen = ArchitectureDiagramGenerator(output_dir)
arch_gen.generate_side_by_side_comparison(standard_config, wave_config)

# Generate loss landscape
loss_viz = LossLandscapeVisualizer(output_dir)
loss_viz.generate_3d_landscape(standard_trajectory, wave_trajectory)

# Generate frequency spectrum
freq_viz = FrequencySpectrumVisualizer(output_dir)
freq_viz.generate_spectrum_evolution(frequency_data)
```

## ArchitectureDiagramGenerator

### Features

- Side-by-side architecture comparisons
- Color-coded layer types
- Parameter count breakdowns
- Animated GIFs showing forward pass

### Methods

#### `generate_side_by_side_comparison(standard_config, wave_config, save_path=None)`

Creates a side-by-side comparison of Standard Transformer and Spectral GPT architectures.

**Parameters:**
- `standard_config` (dict): Configuration for standard transformer
  - `vocab_size`: Vocabulary size
  - `d_model`: Model dimension
  - `num_layers`: Number of transformer layers
  - `num_heads`: Number of attention heads
  - `total_params`: Total parameters in millions
- `wave_config` (dict): Configuration for Spectral GPT (includes `num_waves`, `num_harmonics`)
- `save_path` (str, optional): Path to save figure

**Returns:** Path to saved figure

**Example:**
```python
standard_config = {
    'vocab_size': 50257,
    'd_model': 768,
    'num_layers': 12,
    'num_heads': 12,
    'total_params': 52.9
}

wave_config = {
    'vocab_size': 50257,
    'd_model': 768,
    'num_layers': 12,
    'num_heads': 12,
    'num_waves': 8,
    'num_harmonics': 3,
    'total_params': 67.5
}

generator = ArchitectureDiagramGenerator("output/")
path = generator.generate_side_by_side_comparison(standard_config, wave_config)
```

#### `generate_parameter_breakdown(standard_config, wave_config, save_path=None)`

Creates pie charts showing parameter distribution across components.

**Returns:** Path to saved figure

#### `generate_forward_pass_animation(arch_type='wave', save_path=None, duration=5)`

Generates an animated GIF showing data flow through the architecture.

**Parameters:**
- `arch_type` (str): 'standard' or 'wave'
- `save_path` (str, optional): Path to save GIF
- `duration` (int): Animation duration in seconds

**Returns:** Path to saved GIF

## LossLandscapeVisualizer

### Features

- 3D loss landscape plots
- Optimization trajectory comparisons
- 2D contour projections
- Convergence point highlighting

### Methods

#### `generate_3d_landscape(standard_trajectory, wave_trajectory, save_path=None)`

Creates a 3D visualization of the loss landscape with optimization paths.

**Parameters:**
- `standard_trajectory` (list): List of dicts with 'step' and 'loss' keys
- `wave_trajectory` (list): List of dicts with 'step' and 'loss' keys
- `save_path` (str, optional): Path to save figure

**Returns:** Path to saved figure

**Example:**
```python
# Training trajectories
standard_trajectory = [
    {'step': 0, 'loss': 8.0},
    {'step': 1000, 'loss': 6.5},
    {'step': 2000, 'loss': 5.2},
    # ...
]

wave_trajectory = [
    {'step': 0, 'loss': 7.9},
    {'step': 1000, 'loss': 6.4},
    # ...
]

visualizer = LossLandscapeVisualizer("output/")
path = visualizer.generate_3d_landscape(standard_trajectory, wave_trajectory)
```

#### `generate_contour_plot(standard_trajectory, wave_trajectory, save_path=None)`

Creates a 2D contour plot of the loss landscape.

**Returns:** Path to saved figure

#### `generate_convergence_comparison(standard_trajectory, wave_trajectory, save_path=None)`

Generates convergence trajectory comparison with loss difference plot.

**Returns:** Path to saved figure

## FrequencySpectrumVisualizer

### Features

- Frequency spectrum evolution during training
- Harmonic amplitude distributions
- Phase distribution visualizations
- Wave interference patterns

### Methods

#### `generate_spectrum_evolution(frequency_data, save_path=None)`

Visualizes how frequency distributions evolve during training.

**Parameters:**
- `frequency_data` (dict): Maps step -> frequency array (tokens, waves)
- `save_path` (str, optional): Path to save figure

**Returns:** Path to saved figure

**Example:**
```python
# Frequency data at different training steps
frequency_data = {
    1000: np.random.rand(200, 8),
    5000: np.random.rand(200, 8),
    10000: np.random.rand(200, 8),
}

visualizer = FrequencySpectrumVisualizer("output/")
path = visualizer.generate_spectrum_evolution(frequency_data)
```

#### `generate_harmonic_amplitude_plot(harmonic_data, save_path=None)`

Visualizes harmonic amplitude distributions.

**Parameters:**
- `harmonic_data` (np.ndarray): Array of shape (tokens, waves, harmonics)
- `save_path` (str, optional): Path to save figure

**Returns:** Path to saved figure

#### `generate_phase_distribution(phase_data, save_path=None)`

Creates phase distribution visualizations including polar plots.

**Parameters:**
- `phase_data` (np.ndarray): Array of shape (tokens, waves) with phase values in [0, 2π]
- `save_path` (str, optional): Path to save figure

**Returns:** Path to saved figure

#### `generate_interference_patterns(phase_data, save_path=None)`

Visualizes wave interference patterns between token pairs.

**Parameters:**
- `phase_data` (np.ndarray): Array of shape (tokens, waves) with phase values
- `save_path` (str, optional): Path to save figure

**Returns:** Path to saved figure

#### `generate_frequency_heatmap(frequency_data, token_labels=None, save_path=None)`

Creates a heatmap of frequencies across tokens and waves.

**Parameters:**
- `frequency_data` (np.ndarray): Array of shape (tokens, waves)
- `token_labels` (list, optional): List of token labels for x-axis
- `save_path` (str, optional): Path to save figure

**Returns:** Path to saved figure

## Demo Script

Run the demo script to see all visualizations in action:

```bash
python demo/demo_paper_visualizations.py
```

This will generate all visualization types and save them to `experiments/paper/figures/`.

## Output Files

The generators create the following files:

### Architecture Diagrams
- `architecture_comparison.png`: Side-by-side architecture comparison
- `parameter_breakdown.png`: Parameter distribution pie charts
- `standard_forward_pass.gif`: Animated forward pass (standard)
- `wave_forward_pass.gif`: Animated forward pass (wave)

### Loss Landscapes
- `loss_landscape_3d.png`: 3D loss landscape with trajectories
- `loss_landscape_contour.png`: 2D contour plot
- `convergence_comparison.png`: Convergence trajectory comparison

### Frequency Spectra
- `frequency_spectrum_evolution.png`: Frequency evolution over training
- `harmonic_amplitudes.png`: Harmonic amplitude distributions
- `phase_distribution.png`: Phase distributions with polar plots
- `interference_patterns.png`: Wave interference examples
- `frequency_heatmap.png`: Frequency heatmap across tokens

## Styling

All visualizations use a consistent dark theme with:
- Background: `#1a1a1a`
- Text: White
- Color scheme: Cyan, Yellow, Magenta, Plasma, Viridis

## Integration with Paper Generator

These visualizations can be integrated with the `PaperGenerator` class:

```python
from spectral_gpt.paper_generator import SpectralGPTPaperGenerator
from spectral_gpt.paper_visualizations import *

# Generate visualizations
arch_gen = ArchitectureDiagramGenerator("experiments/paper/figures")
arch_gen.generate_side_by_side_comparison(std_config, wave_config)

# Generate paper with figures
paper_gen = SpectralGPTPaperGenerator("experiments/paper")
paper_path = paper_gen.generate_technical_paper(experiments)
```

## Testing

Run the test suite:

```bash
pytest tests/test_paper_visualizations.py -v
```

All tests should pass, generating temporary visualizations to verify functionality.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- PyTorch (for model integration)

## License

This module is part of the Spectral GPT project.
