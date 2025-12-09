"""
Demo script for paper visualization generators.

This script demonstrates how to use the visualization generators
to create figures for academic papers.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_gpt.paper_visualizations import (
    ArchitectureDiagramGenerator,
    LossLandscapeVisualizer,
    FrequencySpectrumVisualizer
)


def demo_architecture_diagrams():
    """Demonstrate architecture diagram generation"""
    print("\n" + "="*60)
    print("ARCHITECTURE DIAGRAM GENERATION")
    print("="*60)
    
    output_dir = "experiments/paper/figures"
    generator = ArchitectureDiagramGenerator(output_dir)
    
    # Configuration for standard transformer
    standard_config = {
        'vocab_size': 50257,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'total_params': 52.9
    }
    
    # Configuration for Spectral GPT
    wave_config = {
        'vocab_size': 50257,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'num_waves': 8,
        'num_harmonics': 3,
        'total_params': 67.5
    }
    
    # Generate side-by-side comparison
    print("\n1. Generating side-by-side architecture comparison...")
    path = generator.generate_side_by_side_comparison(standard_config, wave_config)
    print(f"   Saved to: {path}")
    
    # Generate parameter breakdown
    print("\n2. Generating parameter breakdown...")
    path = generator.generate_parameter_breakdown(standard_config, wave_config)
    print(f"   Saved to: {path}")
    
    # Generate forward pass animation (optional - takes longer)
    print("\n3. Generating forward pass animations...")
    print("   (This may take a minute...)")
    path_std = generator.generate_forward_pass_animation('standard')
    print(f"   Standard: {path_std}")
    path_wave = generator.generate_forward_pass_animation('wave')
    print(f"   Wave: {path_wave}")


def demo_loss_landscape():
    """Demonstrate loss landscape visualization"""
    print("\n" + "="*60)
    print("LOSS LANDSCAPE VISUALIZATION")
    print("="*60)
    
    output_dir = "experiments/paper/figures"
    visualizer = LossLandscapeVisualizer(output_dir)
    
    # Generate sample training trajectories
    print("\nGenerating sample training trajectories...")
    steps = list(range(0, 15000, 500))
    
    # Standard transformer trajectory
    std_losses = [8.0 * np.exp(-s / 5000) + 4.4 + np.random.normal(0, 0.02) 
                  for s in steps]
    standard_trajectory = [{'step': s, 'loss': l} for s, l in zip(steps, std_losses)]
    
    # Spectral GPT trajectory (slightly different path)
    wave_losses = [7.9 * np.exp(-s / 5500) + 4.48 + np.random.normal(0, 0.03) 
                   for s in steps]
    wave_trajectory = [{'step': s, 'loss': l} for s, l in zip(steps, wave_losses)]
    
    # Generate 3D landscape
    print("\n1. Generating 3D loss landscape...")
    path = visualizer.generate_3d_landscape(standard_trajectory, wave_trajectory)
    print(f"   Saved to: {path}")
    
    # Generate contour plot
    print("\n2. Generating contour plot...")
    path = visualizer.generate_contour_plot(standard_trajectory, wave_trajectory)
    print(f"   Saved to: {path}")
    
    # Generate convergence comparison
    print("\n3. Generating convergence comparison...")
    path = visualizer.generate_convergence_comparison(standard_trajectory, wave_trajectory)
    print(f"   Saved to: {path}")


def demo_frequency_spectrum():
    """Demonstrate frequency spectrum visualization"""
    print("\n" + "="*60)
    print("FREQUENCY SPECTRUM VISUALIZATION")
    print("="*60)
    
    output_dir = "experiments/paper/figures"
    visualizer = FrequencySpectrumVisualizer(output_dir)
    
    # Generate sample frequency data
    print("\nGenerating sample frequency data...")
    
    # Frequency evolution over training
    frequency_data = {}
    for step in [1000, 3000, 5000, 7000, 10000, 15000]:
        # Simulate frequency evolution (more spread and higher frequencies over time)
        alpha = 2 + step / 5000  # Increases over time
        beta = 5 - step / 5000   # Decreases over time
        freqs = np.random.beta(alpha, beta, size=(200, 8))
        frequency_data[step] = freqs
    
    print("\n1. Generating frequency spectrum evolution...")
    path = visualizer.generate_spectrum_evolution(frequency_data)
    print(f"   Saved to: {path}")
    
    # Harmonic amplitudes
    print("\n2. Generating harmonic amplitude plot...")
    harmonic_data = np.random.exponential(0.5, size=(200, 8, 3))
    # Make first harmonic stronger
    harmonic_data[:, :, 0] *= 2
    path = visualizer.generate_harmonic_amplitude_plot(harmonic_data)
    print(f"   Saved to: {path}")
    
    # Phase distribution
    print("\n3. Generating phase distribution...")
    phase_data = np.random.uniform(0, 2*np.pi, size=(200, 8))
    path = visualizer.generate_phase_distribution(phase_data)
    print(f"   Saved to: {path}")
    
    # Interference patterns
    print("\n4. Generating interference patterns...")
    path = visualizer.generate_interference_patterns(phase_data)
    print(f"   Saved to: {path}")
    
    # Frequency heatmap
    print("\n5. Generating frequency heatmap...")
    freq_heatmap_data = frequency_data[15000]  # Use final step
    path = visualizer.generate_frequency_heatmap(freq_heatmap_data)
    print(f"   Saved to: {path}")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("PAPER VISUALIZATION GENERATORS DEMO")
    print("="*60)
    print("\nThis demo generates various visualizations for academic papers.")
    print("All figures will be saved to: experiments/paper/figures/")
    
    try:
        # Run demos
        demo_architecture_diagrams()
        demo_loss_landscape()
        demo_frequency_spectrum()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("\nAll visualizations have been generated successfully.")
        print("Check the experiments/paper/figures/ directory for the output.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
