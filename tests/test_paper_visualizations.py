"""
Tests for paper visualization generators.
"""

import pytest
import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_gpt.paper_visualizations import (
    ArchitectureDiagramGenerator,
    LossLandscapeVisualizer,
    FrequencySpectrumVisualizer
)


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory"""
    return str(tmp_path / "test_visualizations")


@pytest.fixture
def standard_config():
    """Standard transformer configuration"""
    return {
        'vocab_size': 50257,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'total_params': 52.9
    }


@pytest.fixture
def wave_config():
    """Spectral GPT configuration"""
    return {
        'vocab_size': 50257,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'num_waves': 8,
        'num_harmonics': 3,
        'total_params': 67.5
    }


@pytest.fixture
def sample_trajectory():
    """Generate sample training trajectory"""
    steps = list(range(0, 15000, 500))
    # Simulate loss decay
    losses = [8.0 * np.exp(-s / 5000) + 4.4 for s in steps]
    return [{'step': s, 'loss': l} for s, l in zip(steps, losses)]


def test_architecture_diagram_generator(output_dir, standard_config, wave_config):
    """Test architecture diagram generation"""
    generator = ArchitectureDiagramGenerator(output_dir)
    
    # Generate side-by-side comparison
    path = generator.generate_side_by_side_comparison(standard_config, wave_config)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_parameter_breakdown(output_dir, standard_config, wave_config):
    """Test parameter breakdown visualization"""
    generator = ArchitectureDiagramGenerator(output_dir)
    
    path = generator.generate_parameter_breakdown(standard_config, wave_config)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_loss_landscape_3d(output_dir, sample_trajectory):
    """Test 3D loss landscape generation"""
    visualizer = LossLandscapeVisualizer(output_dir)
    
    # Create slightly different trajectory for wave model
    wave_traj = [{'step': d['step'], 'loss': d['loss'] + 0.05} 
                 for d in sample_trajectory]
    
    path = visualizer.generate_3d_landscape(sample_trajectory, wave_traj)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_contour_plot(output_dir, sample_trajectory):
    """Test contour plot generation"""
    visualizer = LossLandscapeVisualizer(output_dir)
    
    wave_traj = [{'step': d['step'], 'loss': d['loss'] + 0.05} 
                 for d in sample_trajectory]
    
    path = visualizer.generate_contour_plot(sample_trajectory, wave_traj)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_convergence_comparison(output_dir, sample_trajectory):
    """Test convergence comparison plot"""
    visualizer = LossLandscapeVisualizer(output_dir)
    
    wave_traj = [{'step': d['step'], 'loss': d['loss'] + 0.05} 
                 for d in sample_trajectory]
    
    path = visualizer.generate_convergence_comparison(sample_trajectory, wave_traj)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_frequency_spectrum_evolution(output_dir):
    """Test frequency spectrum evolution visualization"""
    visualizer = FrequencySpectrumVisualizer(output_dir)
    
    # Generate sample frequency data at different steps
    frequency_data = {}
    for step in [1000, 3000, 5000, 7000, 10000, 15000]:
        # Simulate frequency evolution (more spread over time)
        freqs = np.random.beta(2, 5, size=(100, 8)) * (1 + step / 15000)
        frequency_data[step] = freqs
    
    path = visualizer.generate_spectrum_evolution(frequency_data)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_harmonic_amplitude_plot(output_dir):
    """Test harmonic amplitude visualization"""
    visualizer = FrequencySpectrumVisualizer(output_dir)
    
    # Generate sample harmonic data (tokens, waves, harmonics)
    harmonic_data = np.random.rand(100, 8, 3)
    
    path = visualizer.generate_harmonic_amplitude_plot(harmonic_data)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_phase_distribution(output_dir):
    """Test phase distribution visualization"""
    visualizer = FrequencySpectrumVisualizer(output_dir)
    
    # Generate sample phase data (tokens, waves)
    phase_data = np.random.uniform(0, 2*np.pi, size=(100, 8))
    
    path = visualizer.generate_phase_distribution(phase_data)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_interference_patterns(output_dir):
    """Test interference pattern visualization"""
    visualizer = FrequencySpectrumVisualizer(output_dir)
    
    # Generate sample phase data
    phase_data = np.random.uniform(0, 2*np.pi, size=(50, 8))
    
    path = visualizer.generate_interference_patterns(phase_data)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_frequency_heatmap(output_dir):
    """Test frequency heatmap visualization"""
    visualizer = FrequencySpectrumVisualizer(output_dir)
    
    # Generate sample frequency data
    frequency_data = np.random.rand(50, 8)
    
    path = visualizer.generate_frequency_heatmap(frequency_data)
    assert os.path.exists(path)
    assert path.endswith('.png')


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
